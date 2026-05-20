# Morphopathway Safe Commit Scope

This file records the intended commit and staging boundary for the new `pyXenium.pathway` morphopathway suite and related evidence handoff material.

## Current Repository State

As of the handoff check on 2026-05-19, `git status --short` showed only:

```text
?? release_downloads/
```

`release_downloads/` is outside the morphopathway scope and should not be staged for this work.

## Stage These Paths

Stage the morphopathway suite, tests, workflow scripts, package evidence, and these handoff notes:

```powershell
git add `
  src/pyXenium/pathway/_morphopathway.py `
  src/pyXenium/pathway/__init__.py `
  src/pyXenium/__init__.py `
  tests/test_morphopathway.py `
  benchmarking/morphopathway_atera
```

Do not use broad staging commands such as `git add .` while `release_downloads/` is present.

## Verification Commands

Run the focused unit tests:

```powershell
$env:PYTHONPATH='D:\GitHub\pyXenium\src'
pytest tests\test_morphopathway.py -q
```

Refresh and validate the reviewer audit without changing the archived package payload:

```powershell
python benchmarking\morphopathway_atera\scripts\make_reviewer_evidence_audit.py benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049
python benchmarking\morphopathway_atera\scripts\validate_brief_communication_package.py benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049 --require-reviewer-audit
python benchmarking\morphopathway_atera\scripts\archive_brief_communication_package.py benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049
```

Expected archive SHA256:

```text
db731299997fbdbadab9d4eccba981dfe49330e457bdc907a93b6b3d0d1acf54
```

## Suggested Commit Message

```text
Add morphopathway H&E-WTA evidence package
```

## Commit Boundary

Include:

- `pyXenium.pathway` morphopathway implementation and public exports.
- Focused morphopathway tests.
- Atera morphopathway scripts and generated final evidence package.
- Reviewer audit, QC report, archive manifest, evidence-to-claim map, and safe commit scope.

Exclude:

- `release_downloads/`
- Any old `naturebiotech_package` outputs as evidence input.
- Unrelated benchmark/autopilot scripts outside `benchmarking/morphopathway_atera/`.
