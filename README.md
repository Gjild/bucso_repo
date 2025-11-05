# bucso — Dual-Conversion BUC Spur Optimization

Python 3.13 + uv.

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
bucso init-stubs examples/
bucso validate examples/config.yaml
bucso optimize examples/config.yaml --out out/
bucso report out/policy.yaml --html out/summary.html
