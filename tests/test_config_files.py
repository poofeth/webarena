import json
from pathlib import Path


def test_reddit_task_ground_truth_urls_match_issue_246() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config_files" / "test.raw.json"
    tasks = {task["task_id"]: task for task in json.loads(config_path.read_text())}

    expected_urls = {
        407: ["__REDDIT__/f/deeplearning/125036/should-i-continue-with-this"],
        408: [
            "__REDDIT__/f/explainlikeimfive/125342/eli5-why-does-pressing-my-palms-against-my-eyes-create-a"
        ],
        584: ["__REDDIT__/f/Karaoke/edit", "__REDDIT__/f/Karaoke/edit"],
    }

    for task_id, expected in expected_urls.items():
        urls = [
            check["url"]
            for check in tasks[task_id]["eval"]["program_html"]
        ]
        assert urls == expected
