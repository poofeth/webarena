import json
from pathlib import Path


def test_issue_133_reported_task_text_typos_are_fixed() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config_files" / "test.raw.json"
    serialized_tasks = json.dumps(json.loads(config_path.read_text()))

    assert "Telll" not in serialized_tasks
    assert "canlled" not in serialized_tasks
    assert "correpong" not in serialized_tasks

    assert "Tell me the grand total of invoice" in serialized_tasks
    assert "most recent cancelled order" in serialized_tasks
    assert "corresponding field" in serialized_tasks
