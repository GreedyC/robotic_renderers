# Cosmos 3 enhancement

Cosmos 3 is an optional generative post-process, not a renderer. It consumes an existing
render or synchronized structural controls and may increase visual realism, but its
output is no longer a deterministic rendering-ground-truth product.

## Remote service

The pinned service uses `nvidia/Cosmos3-Super` (64B), BF16, `vllm==0.28.0`, and a pinned
vLLM-Omni 0.28.0 release commit. The pair follows Omni's matching-release requirement;
vLLM 0.29 is not a compatible independent upgrade. It requires a Linux host with at
least four GPUs and selects exactly
four for this preset. GPUs 0–3 are selected by default; choose a different set before
invoking the app when the desired cards have other indices:

```bash
export ROBOTIC_RENDERERS_COSMOS_GPU_IDS=0,1,2,3
uv run robotic-renderers cosmos setup --host render-host
uv run robotic-renderers cosmos start --host render-host
uv run robotic-renderers cosmos status --host render-host
```

The selected GPUs should share a NUMA node when the host topology allows it. The app
validates that every configured index exists and binds the server to that exact set, but
does not guess a topology-specific selection.

The API must remain loopback-only on the remote machine. Model caches live outside the
repository. Stop the service when it is no longer needed:

```bash
uv run robotic-renderers cosmos stop --host render-host
```

## Conditioning modes

Generate a depth-guided 1280x720 video from a newly rendered Blender keyframe:

```bash
uv run robotic-renderers cosmos enhance --backend blender --mode depth \
  --steps 50 --output outputs/cosmos3_depth.mp4
```

Other inputs and controls:

```bash
uv run robotic-renderers cosmos enhance --mode i2v --input outputs/blender.png
uv run robotic-renderers cosmos enhance --mode v2v --input outputs/blender.mp4
uv run robotic-renderers cosmos enhance --backend opengl --mode seg
uv run robotic-renderers cosmos enhance --backend opengl --mode depth-seg
```

Depth is the default structural control because it retains silhouette, floor contact,
prop layout, and camera motion. Segmentation supplies stable robot, floor, and prop
masks. `depth-seg` combines both. Use `--keep-bundle` when the conditioning inputs must
be audited.

The quality preset uses 50 diffusion steps and the model-native 720p shape. Generation
can take substantially longer than conventional rendering. Keep timeout changes
explicit and monitor GPU memory rather than assuming a quiet process has failed.

## Reproducibility and use

The renderer source, conditioning bundle, prompt, negative prompt, model identifier,
runtime commit, seed, dimensions, and generation settings should accompany any retained
result. Generated output is stochastic and should not replace segmentation, depth, or
other simulator ground truth.

Review the
[`Cosmos3-Super` model card](https://huggingface.co/nvidia/Cosmos3-Super) and
OpenMDW-1.1 terms before use or redistribution.
