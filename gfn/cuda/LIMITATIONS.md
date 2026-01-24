# CUDA KERNEL LIMITATIONS
===============================

## Update Scheme

El kernel `recurrent_manifold_fused.cu` usa un **update scheme fijo similar a Heun**:

```cuda
s_v[i] += eff_dt * (s_force[i] - g);
s_x[i] += eff_dt * s_v[i];
```

**NO respeta `integrator_type`** configurado en `MLayer`.

## Integrators Compatibles

✅ **Soportados:**
-  `heun` - Compatible
- `euler` - Compatible  

❌ **NO soportados:**
- `leapfrog` - Update symplectic diferente
- `rk4` - Requiere 4 stages
- `yoshida` - Update symplectic complejo
- Todos los demás symplectic integrators

## Solución

### Para Benchmarks:
Usar `integrator_type='heun'` al crear modelos:

```python
model = Manifold(
    ...,
    integrator_type='heun',  # NO 'leapfrog'!
    ...
)
```

### Para Fix Permanente:
Deshabilitar fused kernel si integrator != heun/euler:

```python
# En model.py línea ~195:
can_fuse = (not self.use_scan and self.depth > 0 and not collect_christ)

# Agregar check:
integrator_compat = all(layer.integrators[0].name in ['heun', 'euler'] 
                       for layer in self.layers)
can_fuse = can_fuse and integrator_compat
```
