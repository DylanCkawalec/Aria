# aria-engine (Python)

Python bindings for the [Aria](https://github.com/aria-ai/aria) transformer runtime.

The extension is a façade over `aria_engine_backends::runner` — the same code
path the `aria` CLI and the WASM module use. Notebook results equal CLI results
byte for byte for the same config.

## Build

```bash
pip install maturin
cd python
maturin develop          # editable install into the active virtualenv
# or: maturin build --release
```

## Use

```python
import aria

engine = aria.AriaEngine(aria.Config(n_modes=64, latent_dim=32, eps=1.0, seed=42))

state = engine.init()
state = engine.step_phi(state)          # OpticalStep -> Predict -> Match -> Diffuse
assert engine.check(state).all_ok

summary = aria.run(steps=1000, config=aria.Config(schedule="opmd"))
# {'steps': 1000, 't': 250, 'graph_size': 250, 'energy': 1.0, 'invariants_ok': True, ...}

jsonl = aria.run_trace_jsonl(steps=1000)  # identical to `aria run --output trace.jsonl`

# Post-hoc decode (𝔸5 / 𝕃5) — the same seam `aria emit` and the WASM
# `emitIds` use, now reachable from a notebook too.
zs = aria.latents(steps=1000, config=aria.Config(schedule="opmd"))
readout = aria.Readout.seeded_discrete(dim=32, vocab_size=256, temperature=1.0, seed=3)
ids = [readout.decode_id(z) for z in zs]

tokenizer = aria.Tokenizer.train(corpus=open("corpus.txt", "rb").read(), vocab_size=4096)
pieces = [tokenizer.decode_one(i) for i in ids]
```

## API

| Symbol | Purpose |
|---|---|
| `aria.Config(...)` | Runtime config; `Config.from_toml(src)` reads the CLI format |
| `aria.AriaEngine(config)` | `init()`, `apply(state, action)`, `step_phi(state)`, `check(state)` |
| `aria.State` | `t`, `energy`, `prev_res`, `graph_size`, `z`, `psi`, `to_json()` |
| `aria.InvariantReport` | `inv1`–`inv4`, `all_ok`, `failures` |
| `aria.run(steps, config)` | Summary dict for a full run |
| `aria.run_trace_jsonl(steps, config)` | JSONL trace string |
| `aria.latents(steps, config)` | Post-step `z` for every action — the `emit` replay seam |
| `aria.Readout` | `from_file`/`from_bytes`/`seeded_discrete`/`seeded_continuous`; `decode_id`, `probs`, `logits`, `emit`, `to_bytes`, `to_file` |
| `aria.Tokenizer` | `bytes_identity`/`train`/`from_file`/`from_json`; `encode`, `decode_one`, `decode`, `to_json`, `to_file` |
| `aria.actions()` | `["OpticalStep", "Predict", "Match", "Diffuse", "Stutter"]` |

## Test

```bash
cargo build -p aria-engine   # the CLI, for the differential parity test
pytest python/tests
```
