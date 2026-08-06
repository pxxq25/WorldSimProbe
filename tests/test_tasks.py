from worldsimprobe.common.tasks import normalize_task_id, required_video_roles


def test_task_ids_and_video_roles() -> None:
    assert normalize_task_id("Task-1") == "task1"
    assert normalize_task_id("5") == "task5"
    assert required_video_roles("task1") == ("original", "small", "large")
    assert required_video_roles("task4") == ("candidate",)
