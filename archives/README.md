# Historical files and cleanup records

`legacy_work` preserves earlier document-builder scripts, extracted reference
text and old logs. Its scripts may contain old paths; they are reference
material, not active build commands.

`organisation_manifest.json` records the original folder reorganisation.
`cleanup_manifest.json` records the subsequent user-approved deletion of three
completed download ZIPs, the old upload ZIP, the previous source bundle, a cache
file and the unfinished browser download. Their former folders may now be empty.

The installed model, working datasets, current Colab packages and training
reports were retained and verified. Original archive names and report hashes
remain in the model import audit and training report source records.

Use `../data/README.md` for datasets, `../colab/README.md` for current uploads,
and `../models/xlmr_emotion` for the installed trained checkpoint. This archive
is local and ignored by Git.

## 29 October 2026 organisation

This folder was renamed from archive to archives. It also retains:

- model_downloads/: three downloaded model ZIPs verified against installed checkpoints.
- verification/: temporary report PDFs and rendered page previews.
- caches/2026-10-29/: Python and pytest cache snapshots.
- installers/: the Tectonic download ZIP; its executable remains available.

Review documents formerly in this archive were moved to ../Documents for review.
organisation_2026-10-29.json records every move and verification. No files were deleted.
See [Documents for review](../Documents%20for%20review/README.md) for current and previous reports.

Archive contents are local and ignored by Git; only this navigation guide is versioned.
