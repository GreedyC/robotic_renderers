# Security policy

## Supported versions

Security updates are made on the latest tagged release and the `main` branch. Older
releases are not maintained unless a release note explicitly says otherwise.

## Reporting a vulnerability

Please use
[GitHub private vulnerability reporting](https://github.com/tkelestemur/robotic_renderers/security/advisories/new)
and include reproduction steps, affected versions, and expected impact. Do not disclose
the issue publicly until a fix is available.

If private reporting is unavailable, contact the maintainer through the GitHub
repository without publishing exploit details. You should receive an acknowledgement
within seven days and a status update after the report has been assessed.

Never include access tokens, SSH keys, model credentials, private hostnames, or user data
in a report. The Cosmos API is designed to remain bound to loopback on its remote host;
exposing it publicly is not a supported deployment.
