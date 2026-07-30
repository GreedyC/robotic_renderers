# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases
use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Use a published `setup-uv` release tag so GitHub Actions can initialize CI jobs.

### Changed

- Removed the legacy root-level Python compatibility modules. Package imports and the
  installed console commands are now the only supported interfaces.
- Replaced maintainer-specific remote host aliases and hardware fixtures with the
  configurable, neutral `render-host` placeholder.
- Updated the main environment and every renderer runtime to the latest stable package
  releases available on 2026-07-29.

## [0.1.0] - 2026-07-26

### Added

- Deterministic Franka FR3 scene shared by five rendering backends.
- Still, live, video, tiled-comparison, and benchmark workflows.
- MuJoCo OpenGL, MuJoCo Warp, Blender Cycles, NVIDIA ovrtx, and Mitsuba backends.
- Optional Cosmos 3 generative enhancement with structural conditioning.
- Public package, contribution, security, citation, and release documentation.

[Unreleased]: https://github.com/tkelestemur/robotic_renderers/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tkelestemur/robotic_renderers/releases/tag/v0.1.0
