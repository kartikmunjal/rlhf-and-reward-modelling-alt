from pathlib import Path

from scripts.publish_repository_research_note import END, START, render


ROOT = Path(__file__).resolve().parents[1]


def test_research_note_is_generated_from_committed_metrics():
    overview, note = render()
    assert (ROOT / "docs/portfolio_research_note.md").read_text(encoding="utf-8") == note.rstrip() + "\n"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count(START) == 1
    assert readme.count(END) == 1
    assert overview in readme


def test_readme_has_one_document_title_and_labels_legacy_evidence():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    in_fence = False
    titles = []
    for line in readme.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("# "):
            titles.append(line)
    assert titles == ["# RLHF Pipeline: Reward Modeling, Policy Optimization, and Scaling"]
    assert "## Implementation and Extension Catalogue" in readme
    assert "preview tables" in readme
