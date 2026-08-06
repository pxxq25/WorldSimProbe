import numpy as np

from worldsimprobe.evaluation.task4_interaction_grounding.tracker import (
    canonical_object_displacement,
    evaluated_object_name,
    object_points,
    task4_no_contact_score,
    task4_scores,
    task4_subset,
)


def test_task4_official_subset_normalization() -> None:
    assert task4_subset({"task4_subtype": "distractor"}) == "distractor_hallucination"
    assert task4_subset({"failure_type": "distractor_hallucination"}) == "distractor_hallucination"


def test_task4_distractor_score_is_independent_of_target_score() -> None:
    combined, diagnostic, source = task4_scores(
        "distractor_hallucination",
        target_accuracy=0,
        distractor_accuracy=1,
        has_distractor=True,
    )
    assert combined == 0
    assert diagnostic == 1
    assert source == "distractor_trajectory_accuracy"


def test_task4_uses_documented_object_for_each_condition() -> None:
    assert evaluated_object_name("distractor_hallucination") == "distractor"
    assert evaluated_object_name("fake_contact_hallucination") == "target"
    assert evaluated_object_name("proximity_hallucination") == "target"


def test_task4_query_grid_is_three_by_three() -> None:
    points = object_points((128.0, 128.0), radius=10, grid=3)
    assert points.shape == (9, 2)
    assert {tuple(point) for point in points.tolist()} == {
        (118.0, 118.0),
        (128.0, 118.0),
        (138.0, 118.0),
        (118.0, 128.0),
        (128.0, 128.0),
        (138.0, 128.0),
        (118.0, 138.0),
        (128.0, 138.0),
        (138.0, 138.0),
    }


def test_task4_displacement_is_measured_in_canonical_coordinates() -> None:
    result = canonical_object_displacement(
        {
            "trajectory": [[10.0, 10.0], [15.0, 10.0]],
            "source_width": 128,
            "source_height": 128,
        }
    )
    assert np.isclose(result["max_displacement_px_256"], 10.0)


def test_task4_requires_both_motion_gate_and_static_object() -> None:
    assert task4_no_contact_score(10.0, 60.0)["task4_score"] == 100.0
    assert task4_no_contact_score(10.01, 60.0)["task4_score"] == 0.0
    assert task4_no_contact_score(1.0, 59.99)["task4_score"] == 0.0
