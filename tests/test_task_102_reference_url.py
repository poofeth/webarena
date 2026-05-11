import json
from pathlib import Path


def test_task_102_reference_url_matches_requested_repo() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config_files" / "test.raw.json"
    tasks = {task["task_id"]: task for task in json.loads(config_path.read_text())}

    task = tasks[102]

    assert task["instantiation_dict"]["repo"] == "a11yproject/a11yproject.com"
    assert (
        task["eval"]["reference_url"]
        == "__GITLAB__/a11yproject/a11yproject.com/-/issues/?label_name%5B%5D=help%20wanted"
    )
