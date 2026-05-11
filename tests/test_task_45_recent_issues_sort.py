import json
from pathlib import Path


def test_task_45_sorts_open_issues_by_most_recent_first() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config_files" / "test.raw.json"
    tasks = {task["task_id"]: task for task in json.loads(config_path.read_text())}

    assert tasks[45]["intent"] == "Check out the most recent open issues"
    assert (
        tasks[45]["eval"]["reference_url"]
        == "__GITLAB__/a11yproject/a11yproject.com/-/issues/?sort=created_desc&state=opened"
    )
