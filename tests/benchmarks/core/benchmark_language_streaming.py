import argparse
import itertools
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.losses import ToroidalDistanceLoss, geodesic_regularization, hamiltonian_loss
from gfn.optim import RiemannianAdam
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats


class SimpleCharTokenizer:
    def __init__(self, text):
        charset = sorted(set(text))
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.stoi = {ch: i + 2 for i, ch in enumerate(charset)}
        self.itos = {i + 2: ch for i, ch in enumerate(charset)}

    def encode(self, text, add_special_tokens=False):
        ids = [self.stoi.get(ch, self.eos_token_id) for ch in text]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __len__(self):
        return len(self.stoi) + 2


def build_tokenizer(source, fallback_text):
    if source == "fallback":
        return SimpleCharTokenizer(fallback_text), "fallback"
    try:
        from transformers import AutoTokenizer
    except Exception:
        return SimpleCharTokenizer(fallback_text), "fallback"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, "gpt2"


def build_stream(source, split, max_texts, fallback_text):
    if source == "fallback":
        lines = fallback_text.split("\n")
        return itertools.islice(itertools.cycle(lines), max_texts)
    try:
        from datasets import load_dataset
    except Exception:
        lines = fallback_text.split("\n")
        return itertools.islice(itertools.cycle(lines), max_texts)
    if source in {"wikitext", "wikitext-103-v1"}:
        try:
            dataset = load_dataset("wikitext", "wikitext-103-v1", split=split, streaming=True)
        except Exception:
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    elif source == "wikitext-103-raw-v1":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    else:
        dataset = load_dataset(source, split=split, streaming=True)
    return itertools.islice(dataset, max_texts)


class StreamingTokenDataset(IterableDataset):
    def __init__(self, stream_builder, tokenizer, seq_len):
        self.stream_builder = stream_builder
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        stream = self.stream_builder()
        buffer = []
        for item in stream:
            text = item["text"] if isinstance(item, dict) else item
            if not text or not text.strip():
                continue
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_ids.append(self.tokenizer.eos_token_id)
            buffer.extend(token_ids)
            while len(buffer) > self.seq_len:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                src = torch.tensor(chunk[:-1], dtype=torch.long)
                tgt = torch.tensor(chunk[1:], dtype=torch.long)
                yield src, tgt
        if len(buffer) > 1:
            pad_id = self.tokenizer.pad_token_id
            pad_len = (self.seq_len + 1) - len(buffer)
            padded = buffer + [pad_id] * pad_len
            src = torch.tensor(padded[:-1], dtype=torch.long)
            tgt = torch.tensor(padded[1:], dtype=torch.long)
            yield src, tgt


def build_model(vocab_size, dim, depth, heads, integrator, use_scan):
    physics_config = {
        "embedding": {"type": "functional", "mode": "linear", "coord_dim": 16},
        "readout": {"type": "implicit", "coord_dim": 16},
        "active_inference": {
            "enabled": True,
            "dynamic_time": {"enabled": True},
            "reactive_curvature": {"enabled": True, "plasticity": 0.2},
            "singularities": {"enabled": True, "strength": 20.0, "threshold": 0.8},
        },
        "fractal": {"enabled": True, "threshold": 0.5, "alpha": 0.2},
        "topology": {"type": "torus"},
        "stability": {"base_dt": 0.4},
    }
    return Manifold(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        heads=heads,
        integrator_type=integrator,
        use_scan=use_scan,
        physics_config=physics_config,
        impulse_scale=80.0,
        holographic=False,
    )


def extract_logits(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def tokens_to_angles(token_ids, vocab_size):
    two_pi = 2.0 * math.pi
    return (token_ids.float() / max(vocab_size, 1)) * two_pi


def angles_to_token_ids(pred_angles, vocab_size):
    two_pi = 2.0 * math.pi
    scaled = torch.round((pred_angles / two_pi) * vocab_size).long()
    return torch.remainder(scaled, vocab_size)


def compute_holographic_loss(model, output, x_pred, target_angles, mask):
    criterion = ToroidalDistanceLoss()
    if mask is None:
        loss_val = criterion(x_pred, target_angles)
    else:
        valid = mask.unsqueeze(-1).expand_as(x_pred)
        if valid.any():
            loss_val = criterion(x_pred[valid], target_angles[valid])
        else:
            loss_val = x_pred.sum() * 0.0
    loss_phy = 0.0
    loss_ham = 0.0
    if isinstance(output, tuple) and len(output) >= 6:
        christoffels = output[2]
        v_seq = output[3]
        x_seq = output[4]
        all_forces = output[5]
        if christoffels:
            loss_phy = geodesic_regularization(None, christoffels, lambda_g=0.001)

            def first_head_metric(x):
                return (
                    model.layers[0].christoffels[0].get_metric(x)
                    if hasattr(model.layers[0].christoffels[0], "get_metric")
                    else torch.ones_like(x)
                )

            loss_ham = hamiltonian_loss(
                v_seq, states=x_seq, metric_fn=first_head_metric, lambda_h=0.0, forces=all_forces
            )
    return loss_val + loss_phy + loss_ham


def run_training(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    max_steps,
    log_every,
    loss_fn,
    pad_token_id,
    vocab_size,
    use_holographic,
):
    model.train()
    total_loss = 0.0
    data_iter = iter(loader)
    pbar = tqdm(range(1, max_steps + 1), desc="Entrenando", dynamic_ncols=True)
    for step in pbar:
        try:
            src, tgt = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            src, tgt = next(data_iter)
        src = src.to(device)
        tgt = tgt.to(device)
        mask = tgt != pad_token_id
        optimizer.zero_grad()
        output = model(src, collect_christ=False)
        logits = extract_logits(output)
        if use_holographic:
            target_angles = tokens_to_angles(tgt, vocab_size).unsqueeze(-1).expand_as(logits)
            total_loss_val = compute_holographic_loss(model, output, logits, target_angles, mask)
        else:
            total_loss_val = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
        if torch.isnan(total_loss_val):
            continue
        total_loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += total_loss_val.item()
        avg = total_loss / step
        ppl = math.exp(min(avg, 20.0)) if not use_holographic else None
        acc = 0.0
        if logits.dim() == 3:
            if use_holographic:
                pred_angles = logits.mean(dim=-1)
                pred_ids = angles_to_token_ids(pred_angles, vocab_size)
            else:
                pred_ids = logits.argmax(dim=-1)
            if mask.any():
                acc = (pred_ids[mask] == tgt[mask]).float().mean().item()
            else:
                acc = 0.0
        if step % log_every == 0 or step == 1:
            ppl_display = f"{ppl:.2f}" if ppl is not None else "n/a"
            pbar.set_postfix(loss=f"{avg:.4f}", ppl=ppl_display, acc=f"{acc*100:.2f}%")
    return total_loss / max_steps


@torch.no_grad()
def run_evaluation(model, loader, device, max_steps, loss_fn, pad_token_id, vocab_size, use_holographic):
    model.eval()
    total_loss = 0.0
    data_iter = iter(loader)
    for _ in range(max_steps):
        try:
            src, tgt = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            src, tgt = next(data_iter)
        src = src.to(device)
        tgt = tgt.to(device)
        output = model(src, collect_christ=False)
        logits = extract_logits(output)
        if use_holographic:
            mask = tgt != pad_token_id
            target_angles = tokens_to_angles(tgt, vocab_size).unsqueeze(-1).expand_as(logits)
            loss = compute_holographic_loss(model, output, logits, target_angles, mask)
        else:
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
        if not torch.isnan(loss):
            total_loss += loss.item()
    return total_loss / max_steps


def run_language_streaming_test(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fallback_text = "\n".join(
        [
            "the quick brown fox jumps over the lazy dog",
            "manifold models language as a geometric flow",
            "streaming datasets avoid loading everything into ram",
            "this is a tiny fallback corpus for tests",
        ]
    )
    tokenizer, tokenizer_name = build_tokenizer(args.source, fallback_text)
    vocab_size = len(tokenizer)
    train_dataset = StreamingTokenDataset(
        lambda: build_stream(args.source, "train", args.max_texts, fallback_text),
        tokenizer,
        args.seq_len,
    )
    val_dataset = StreamingTokenDataset(
        lambda: build_stream(args.source, "validation", args.max_val_texts, fallback_text),
        tokenizer,
        args.seq_len,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = build_model(
        vocab_size=vocab_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        integrator=args.integrator,
        use_scan=args.use_scan,
    ).to(device)

    base_lr = args.lr
    optimizer = RiemannianAdam(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(x in n for x in ["x0", "v0", "impulse_scale", "gate"])
                ],
                "lr": base_lr,
                "weight_decay": 1e-4,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(x in n for x in ["x0", "v0", "impulse_scale", "gate"])
                ],
                "lr": base_lr * 10.0,
                "weight_decay": 0.0,
            },
        ]
    )
    scheduler = None
    if args.steps >= 20:
        try:
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=base_lr * 10.0, total_steps=args.steps, pct_start=0.2
            )
        except Exception:
            scheduler = None

    logger = ResultsLogger("language_streaming", category="core")
    from gfn.readout import ImplicitReadout

    use_holographic = isinstance(model.readout, ImplicitReadout)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) if not use_holographic else None
    t0 = time.time()
    train_loss = run_training(
        model,
        train_loader,
        optimizer,
        scheduler,
        device,
        args.steps,
        args.log_every,
        loss_fn,
        tokenizer.pad_token_id,
        vocab_size,
        use_holographic,
    )
    train_time = time.time() - t0
    eval_loss = run_evaluation(
        model,
        val_loader,
        device,
        args.eval_steps,
        loss_fn,
        tokenizer.pad_token_id,
        vocab_size,
        use_holographic,
    )
    eval_ppl = math.exp(min(eval_loss, 20.0)) if not use_holographic else None

    save_path = logger.results_dir / "language_streaming_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "tokenizer_name": tokenizer_name,
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "integrator": args.integrator,
            "use_scan": args.use_scan,
        },
        save_path,
    )

    reloaded = build_model(
        vocab_size=vocab_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        integrator=args.integrator,
        use_scan=args.use_scan,
    ).to(device)
    reloaded.load_state_dict(torch.load(save_path, map_location=device)["model_state_dict"])
    reload_eval_loss = run_evaluation(
        reloaded,
        val_loader,
        device,
        args.eval_steps,
        loss_fn,
        tokenizer.pad_token_id,
        vocab_size,
        use_holographic,
    )

    def _eval_call():
        run_evaluation(
            reloaded,
            val_loader,
            device,
            min(5, args.eval_steps),
            loss_fn,
            tokenizer.pad_token_id,
            vocab_size,
            use_holographic,
        )

    peak_mem = PerformanceStats.measure_peak_memory(reloaded, _eval_call)

    metrics = {
        "source": args.source,
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "eval_ppl": eval_ppl,
        "reload_eval_loss": reload_eval_loss,
        "train_time_sec": train_time,
        "peak_vram_mb": peak_mem,
        "steps": args.steps,
        "eval_steps": args.eval_steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "max_texts": args.max_texts,
        "max_val_texts": args.max_val_texts,
    }
    logger.save_json(metrics)
    print(f"Model saved to: {save_path}")
    if eval_ppl is None:
        print(f"Eval Loss: {eval_loss:.4f} | PPL: n/a | Reload Eval Loss: {reload_eval_loss:.4f}")
    else:
        print(
            f"Eval Loss: {eval_loss:.4f} | PPL: {eval_ppl:.2f} | Reload Eval Loss: {reload_eval_loss:.4f}"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="wikitext")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--max-texts", type=int, default=1000)
    parser.add_argument("--max-val-texts", type=int, default=200)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--integrator", type=str, default="leapfrog")
    parser.add_argument("--use-scan", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_language_streaming_test(args)
