# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases
use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Refresh the main and isolated renderer dependency locks, build tools, and CI actions.
- Upgrade the throughput harness to Blender 5.2.1 LTS with its official archive checksum.
- Upgrade Cosmos to the vLLM-Omni 0.28.0 release commit and matching vLLM 0.28.0.

### Fixed

- Read ovrtx 0.5 RGB and AOV outputs by full USD path, including throughput workers,
  and migrate the retired specular/transmission bounce attributes.
- Resolve the Cosmos package pair together and reject stale existing installations.
- Let Blender throughput workers import the project through the supplied `PYTHONPATH`
  and report early worker exits instead of waiting for the readiness deadline.

## [0.1.0] - 2026-07-31

### Added

- Deterministic Franka FR3 scene shared by five rendering backends.
- Still, live, video, tiled-comparison, and benchmark workflows.
- MuJoCo OpenGL, MuJoCo Warp, Blender Cycles, NVIDIA ovrtx, and Mitsuba backends.
- Optional Cosmos 3 generative enhancement with structural conditioning.
- Public package, contribution, security, citation, and release documentation.

### Changed

- Removed the legacy root-level Python compatibility modules. Package imports and the
  installed console commands are now the only supported interfaces.
- Replaced maintainer-specific remote host aliases and hardware fixtures with the
  configurable, neutral `render-host` placeholder.
- Updated the main environment and every renderer runtime to the latest stable package
  releases available on 2026-07-29.

### Fixed

- Use a published `setup-uv` release tag so GitHub Actions can initialize CI jobs.

[Unreleased]: https://github.com/tkelestemur/robotic_renderers/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tkelestemur/robotic_renderers/releases/tag/v0.1.0
