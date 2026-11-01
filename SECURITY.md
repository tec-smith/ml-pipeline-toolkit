# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[security advisory form](https://github.com/tec-smith/ml-pipeline-toolkit/security/advisories/new),
not in a public issue. Include steps to reproduce and the impact you expect.
You'll get an acknowledgement within a few days.

## Scope

This policy covers the toolkit itself. If you've built a system with it, you
are responsible for that system's security, including protecting the
prediction API, which has no authentication of its own, and the storage your
model bundles are loaded from.

Model bundles contain a Python pickle, and loading a pickle can run code. A
bundle from an untrusted source being able to run code when loaded is
expected behaviour, not a vulnerability. Please do report a way to make the
toolkit load a file other than the one it was given, or to get past the
checksum check without changing `metadata.json`.

See the README's [security section](README.md#security) and
[production checklist](README.md#before-going-to-production).
