# Backend parity

The canonical scene is assembled in `robotic_renderers.scene`. Renderer differences
should reflect quality and transport, not different inputs.

## Shared scene

Every backend receives:

- the same floor-mounted Franka FR3 and joint trajectory;
- one box, sphere, and cylinder plus two genuine mesh props;
- 0.33 m checker tiles and the same floor reflectance intent;
- a 44-degree vertical field of view at 1280x720 by default;
- the same deterministic 30-second, two-orbit camera path;
- the same gradient sky;
- a directional daylight key, cool point fill, and warm spot rim.

The light positions, directions, colors, cone angles, and material parameters originate
in the canonical scene. Translators convert their intensities to each renderer's units.

## Intentional renderer-specific behavior

| Backend | Transport and output response |
| --- | --- |
| MuJoCo OpenGL | Phong rasterization with shadows, reflection, skybox, haze, and MSAA. A documented output gamma lifts its shadow-heavy framebuffer response without changing authored lights. |
| MuJoCo Warp | Direct-light ray rendering with textures, shadows, ambient/direct lighting, sky, specular response, and configurable supersampling. It does not provide multibounce global illumination. |
| Blender | Cycles path tracing for final output and an interactive viewport for live use. Canonical directional/point/spot lights map to Blender Sun/Point/Spot lights. AgX and a fixed exposure offset match the common perceived level. |
| ovrtx | Explicit `pathtraced` and `realtime` modes. Path-traced RGB remains the final color product when synchronized real-time AOVs are requested. Physical camera exposure and renderer-unit light conversion are fixed in the exporter. |
| Mitsuba | CUDA/OptiX path tracing with multiple importance sampling, 12-bounce transport, AOV-guided denoising, and a fixed filmic tone response. |

Blender reads every light's type, transform, diffuse color, range, cone, and shadow flag
from the shared manifest. Its only light-specific compensation converts MuJoCo output
and apparent source size to Blender units: directional/point/spot energy scales are
1x/10x/20x, point/spot bulb-radius scales are 1x/4x, and directional lights use a
3-degree Sun angle for soft studio shadows.

Mitsuba likewise translates every canonical light from the manifest. Mitsuba 3.9 does
not expose a supported per-emitter switch for disabling shadow-ray visibility, so the
canonical fill light's `cast_shadow=false` intent is retained in runtime metadata but
the path tracer still evaluates its visibility. This is an explicit backend limitation,
not a scene-content change.

These are rendering-pipeline differences. Do not tune quality by changing object
placement, tile cadence, camera motion, lens, or light layout in only one backend.

## AOV synchronization

ovrtx supports normalized `depth`, `normals`, `albedo`, `semantic`, and `instance`
outputs. The path-traced color product does not expose all annotations, so AOV requests
use a synchronized real-time annotation product while preserving converged path-traced
RGB. Instance IDs are stable derivations from unique USD label semantics.

Depth arrays remain metric. Preview images and videos use one fixed visualization range
for the whole sequence so apparent brightness cannot shift from frame to frame.

## Reviewing parity changes

For any translator change:

1. render the same time sample, dimensions, and camera in every affected backend;
2. compare geometry silhouettes, floor contact, framing, tile cadence, and light
   direction before judging aesthetic quality;
3. document any new exposure or color-response compensation;
4. run an individual smoke test and `--backend all` for orchestration changes.
