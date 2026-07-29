# Cosmos 3 runtime

Cosmos runs only on the configured Linux multi-GPU host. Its runtime is
recorded in `manifest.toml` rather than a checked-in `uv.lock` because the
environment combines CUDA-specific PyTorch resolution, vLLM, a Git-pinned
vLLM-Omni checkout, and gated model artifacts. A lock generated on macOS or
without the intended CUDA package index would not faithfully describe that
environment.

The installation sequence used on the remote host is:

```bash
uv venv --python 3.12 --managed-python --seed .venv
uv pip install --python .venv/bin/python --torch-backend=cu130 \
  'vllm==0.26.0'
uv pip install --python .venv/bin/python --torch-backend=cu130 \
  'vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@39663d52872019fb0eb74bed8f1e811190486b0f'
```

The model must be loaded from the exact Hugging Face revision recorded in the
manifest:

```text
nvidia/Cosmos3-Super@e0262be9d8f7586bc24c069a2aed2b665bdff266
```

Model weights, Hugging Face credentials, caches, and generated environments
must remain outside Git. The service binds to loopback and reads its API key
from private remote state managed by the application.
