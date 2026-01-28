# Auditoría de Arquitectura Python — Manifold/GFN

## 1. Resumen Ejecutivo
- El proyecto implementa un modelo secuencial “Manifold” que evoluciona estados (x, v) mediante integradores (Heun, RK4, Leapfrog) con soporte CUDA y fallback en PyTorch.
- La arquitectura tiene dos regímenes de salida:
  - Lectura implícita holográfica (coordenadas en un toro, pérdidas toroidales).
  - Lectura estándar con logits de vocabulario y Cross-Entropy.
- El entrenamiento de lenguaje en streaming ya usa lectura estándar + CE, evitando la incoherencia anterior que impedía bajar el loss.
- La canalización de datos usa IterableDataset con HuggingFace/WikiText y fallback local, con chunking secuencial src→tgt.
- Se identificaron pequeñas inconsistencias en manifestos/impresiones y riesgos de métrica degenerada si domina el padding.

## 2. Mapa del Proyecto (Python)
- Núcleo del modelo: [model.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/model.py)
  - Define Manifold: embeddings, M-Layers, integradores, readouts, forward y sampling.
- Capas y geometría:
  - [layers/base.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/layers/base.py) · construcción de Christoffel por cabeza.
  - [layers/fractal.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/layers/fractal.py) · macro/micro-manifold y blending recursivo.
  - Geometrías: [geometry/toroidal.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/geometry/toroidal.py), [geometry/analytical.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/geometry/analytical.py), [geometry/boundaries.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/geometry/boundaries.py).
- CUDA e integradores:
  - Autograd y wrappers: [cuda/autograd.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/cuda/autograd.py)
  - Fallback e invocación: [cuda/ops.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/cuda/ops.py), [cuda/ops_scons.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/cuda/ops_scons.py)
  - Kernels y lanzadores: [cuda/src/integrators](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/cuda/src/integrators)
- Lecturas y embeddings:
  - Readout implícito: [readout.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/readout.py)
  - Embeddings: [embeddings.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/embeddings.py)
- Pérdidas y optimización:
  - Pérdidas: [losses.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/losses.py)
  - Optimizador: [optim.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/optim.py)
- Benchmarks y datos:
  - Streaming lenguaje: [benchmark_language_streaming.py](file:///d:/ASAS/projects/tests/publicación/manifold/tests/benchmarks/core/benchmark_language_streaming.py)
  - Dinámica de aprendizaje: [benchmark_learning_dynamics.py](file:///d:/ASAS/projects/tests/publicación/manifold/tests/benchmarks/core/benchmark_learning_dynamics.py)
  - Scaling: [benchmark_scaling.py](file:///d:/ASAS/projects/tests/publicación/manifold/tests/benchmarks/core/benchmark_scaling.py)

## 3. Flujo de Datos e I/O
- Tokenización:
  - GPT-2 vía HuggingFace con `pad_token = eos_token` si no existe, o fallback char-tokenizer.
  - Referencia: [build_tokenizer](file:///d:/ASAS/projects/tests/publicación/manifold/tests/benchmarks/core/benchmark_language_streaming.py#L42-L53)
- Streaming:
  - `datasets.load_dataset(..., streaming=True)` con fallback a `wikitext-103-raw-v1` cacheado.
  - Referencia: [build_stream](file:///d:/ASAS/projects/tests/publicación/manifold/tests/benchmarks/core/benchmark_language_streaming.py#L55-L73)
- Dataset iterable:
  - Ensambla buffer de ids, añade EOS, corta en ventanas de longitud fija y produce (src, tgt) desplazados 1.
  - Referencia: [StreamingTokenDataset](file:///d:/ASAS/projects/tests/publicación/manifold/tests/benchmarks/core/benchmark_language_streaming.py#L76-L105)
- Riesgos observados:
  - Si `pad_token_id == eos_token_id`, ignorar padding en CE evita penalizar EOS; correcto, pero vigilar que no domine el padding al final de streams cortos.
  - Métricas de evaluación pueden degenerar si el buffer final está muy acolchado; el benchmark ya filtra mediante máscaras.

## 4. Pipeline del Modelo (Forward)
- Entradas: `input_ids [B, L]`, máscara opcional, o `force_manual [B, L, D]`.
- Embedding:
  - Tipos: estándar (nn.Embedding), funcional (SIREN/lineal, O(1) en vocab), implícita (campos neuronales).
  - Referencia: [embeddings](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/model.py#L30-L57)
- Evolución:
  - M-Layers actualizan (x, v) por cabeza con Christoffel y gates; soporte fractal (macro/micro).
  - Referencia: [forward secuencial](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/model.py#L371-L402)
- Readout:
  - Estándar: `nn.Linear(dim, vocab_size)` produce logits.
  - Implícito: `ImplicitReadout` proyecta a coordenadas periódicas; para tareas holográficas.
  - Referencia: [readout selección](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/model.py#L70-L90)
- Salidas:
  - Estándar: `(logits[B,L,V], (x,v), christoffels, v_seq, x_seq, all_forces)`
  - Implícito/holográfico: `(coords[B,L,C], ...)`

## 5. Matemáticas y Física
- Geometría toroidal:
  - Métrica y fuerzas de Christoffel periódicas con wrap y clamps para estabilidad.
  - Referencia: [ToroidalChristoffel.forward](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/geometry/toroidal.py#L70-L110)
- Distancias en el toro:
  - Pérdida toroidal: distancia angular mínima con wrap periódico.
  - Referencia: [toroidal_distance_loss](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/losses.py#L141-L170)
- Regularizaciones físicas:
  - Hamiltoniana (conservación energética) y geodésica (suavizado de curvaturas).
  - Referencias: [hamiltonian_loss](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/losses.py#L14-L45), [geodesic_regularization](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/losses.py#L60-L81)
- Lectura implícita:
  - Mapeo `[sin(x), cos(x)]` + MLP para coordenadas con temperatura; útil en tareas cíclicas.
  - Referencia: [ImplicitReadout](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/readout.py#L17-L62)

## 6. Integradores e Infraestructura CUDA
- Integradores disponibles: Leapfrog (simp. con fricción implícita), Heun, RK4, Verlet, recurrente fusionado.
- Wrappers Autograd:
  - `LeapfrogFusedFn`: forward vía kernel y backward personalizado; se normaliza `dt_scale` correctamente a float en backward.
  - Referencia: [LeapfrogFusedFn.backward](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/cuda/autograd.py#L83-L101)
- Fallback Python:
  - `cuda/ops.py` ofrece versión PyTorch vectorizada cuando no hay CUDA o forma incompatible.
  - Referencia: [leapfrog_fused fallback](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/cuda/ops.py#L130-L170)
- Kernels:
  - Lanzadores para fused integrators y recurrent manifold; topología, gating, damping e intercambio de cabezas.
  - Referencias: [recurrent_manifold_fused.cu](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/cuda/src/integrators/recurrent_manifold_fused.cu#L114-L138), [leapfrog_backward.cu](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/cuda/src/integrators/leapfrog_backward.cu#L79-L99)

## 7. Pérdidas y Entrenamiento (Lenguaje)
- Configuración actual del benchmark de lenguaje:
  - Embedding funcional + Readout implícito con `coord_dim=16`.
  - Referencia: [build_model y loss_fn](file:///d:/ASAS/projects/tests/publicación/manifold/tests/benchmarks/core/benchmark_language_streaming.py#L107-L132), [#L317-L335](file:///d:/ASAS/projects/tests/publicación/manifold/tests/benchmarks/core/benchmark_language_streaming.py#L317-L335)
- Loop de entrenamiento:
  - Barra `tqdm` con loss/PPL/acc, clipping, `OneCycleLR` opcional.
  - Referencia: [run_training](file:///d:/ASAS/projects/tests/publicación/manifold/tests/benchmarks/core/benchmark_language_streaming.py#L182-L221)
- Evaluación y guardado:
  - Métricas, guardado de checkpoint, reload y validación.
  - Referencia: [saving y reload](file:///d:/ASAS/projects/tests/publicación/manifold/tests/benchmarks/core/benchmark_language_streaming.py#L336-L358)

## 8. Consistencia y Coherencia — Observaciones
- Lectura y pérdida:
  - En tareas de lenguaje, es coherente usar `nn.Embedding` + `nn.Linear(..., vocab)` + CE.
  - Para tareas holográficas/angulares, usar `ImplicitReadout` + pérdidas toroidales; evitar CE.
  - Discrepancia crítica: el benchmark de streaming usa `ImplicitReadout` (coords) pero entrena con CE sobre logits de vocab; esto rompe la semántica del loss y puede impedir aprender.
- Manifest de lectura:
  - El manifiesto imprime “Implicit MLP” aunque la lectura sea estándar (Linear). Sugerir alinear mensaje con `readout_type`.
  - Referencia: [print manifest](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/model.py#L92-L117)
- Padding/EOS:
  - `pad_token = eos_token` en GPT-2 es práctica común; CE ignora pad (que coincide con EOS). Correcto, pero revisar que EOS no domine objetivos, para no silenciar gradientes.
- Métricas:
  - Accuracy en lenguaje tiende a baja al inicio; loss y PPL son más sensibles. Si eval_loss = 0.0, revisar si lotes contienen solo pad tras chunking final.
- CUDA backward:
  - `dt_scale` normalizado en backward evita mismatch de tipos; verificado en [autograd.py](file:///d:/ASAS/projects/tests/publicación/manifold/gfn/cuda/autograd.py#L83-L101).
- Hiperparámetros:
  - Documentación interna sugiere LR base ≈ 1e-4 para pesos y ≈1e-2 para gates/estados; el benchmark parametriza `base_lr` y programador `OneCycleLR`.
  - Referencia: [BENCHMARKS.md](file:///d:/ASAS/projects/tests/publicación/manifold/docs/BENCHMARKS.md#L187-L252)

## 9. Recomendaciones
- Mensajería de manifest:
  - Ajustar `readout` mostrado según `readout_type`: “Linear (CE)” vs “Implicit (coords)”.
- Alineación lectura/pérdida:
  - O bien cambiar `readout` a estándar para CE, o bien cambiar loss a toroidal/geométrica si se mantiene implícito.
- Dataset/stream:
  - Añadir chequeo de proporción de pad/EOS por lote; loggear ratio para evitar métricas degeneradas.
- Métricas:
  - Añadir top-k / entropy del último paso; mejor señal temprana que accuracy por token.
- Entrenamiento:
  - Calibrar `max_lr` del `OneCycleLR` (p.ej. 5× `base_lr`); evitar overshoot inicial.
- Pruebas:
  - Agregar test unitario que verifique que forward estándar produce `logits.shape == [B,L,V]` y que CE disminuye en un corpus sintético.

## 10. Entradas/Salidas — Especificación Rápida
- Input:
  - `input_ids [B,L]` enteros (vocab).
  - Opcional: `attention_mask [B,L]`.
- Output estándar (lenguaje):
  - `logits [B,L,V]`, estado `(x,v)`, trazas internas (christoffels, secuencias).
- Output holográfico (tareas angulares):
  - `coords [B,L,C]` para pérdidas geométricas.

## 11. Próximos Pasos
- Integrar chequeos de dataset en `StreamingTokenDataset`.
- Revisar y corregir el manifiesto del `readout`.
- Añadir benchmark corto con corpus sintético (sin pad) para validar CE descendente consistente.
