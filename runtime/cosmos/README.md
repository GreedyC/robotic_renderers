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
  'vllm==0.28.0' \
  'vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c'
```

This is the vLLM-Omni 0.28.0 release commit. Its supported vLLM release line is
0.28.x, so vLLM 0.29.0 is intentionally excluded. Both packages are resolved in
one transaction to prevent the second installation from replacing the first
package's dependencies with an incompatible set. Existing services must be stopped
before `cosmos setup --force` recreates their environment. Setup checks the installed
vLLM version and Omni Git commit instead of accepting any existing `vllm` executable.

The model must be loaded from the exact Hugging Face revision recorded in the
manifest:

```text
nvidia/Cosmos3-Super@e0262be9d8f7586bc24c069a2aed2b665bdff266
```

Model weights, Hugging Face credentials, caches, and generated environments
must remain outside Git. The service binds to loopback and reads its API key
from private remote state managed by the application.
