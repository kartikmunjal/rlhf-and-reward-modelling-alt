# README rollback record

The complete README from immediately before the 2026-11-19 provenance cleanup
is preserved by the annotated Git tag:

```text
pre-readme-cleanup-2026-11-19
```

View that README without modifying the working tree:

```bash
git show pre-readme-cleanup-2026-11-19:README.md
```

Restore only the old README into the working tree for review:

```bash
git restore --source pre-readme-cleanup-2026-11-19 -- README.md
```

The rollback version contains historical expected, example, simulated, and
noncanonical figures that were removed from the current README. Restoring it
does not convert those figures into measured evidence. The current generated
README and its linked metrics artifacts remain the authoritative research
record unless a rollback is explicitly committed.

Checkpoint created from commit `dd130f5` before cleanup commit `41f15da`.
