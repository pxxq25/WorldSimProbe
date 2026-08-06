import pytest

from worldsimprobe.evaluation.task1_action_calibration.oracle_ratio import (
    score_task1_oracle_ratio,
    triangular_ratio_score,
)


def test_triangular_ratio_peaks_at_oracle() -> None:
    assert triangular_ratio_score(0.25, 0.25) == pytest.approx(1.0)
    assert triangular_ratio_score(0.125, 0.25) == pytest.approx(0.5)
    assert triangular_ratio_score(0.625, 0.25) == pytest.approx(0.5)


def test_task1_score_uses_simulator_ratio() -> None:
    result = score_task1_oracle_ratio(
        small_mse=2.0,
        large_mse=8.0,
        simulator_small_mse=1.0,
        simulator_large_mse=4.0,
    )
    assert result["oracle_valid"]
    assert result["direction_pass"]
    assert result["oracle_ratio_score"] == pytest.approx(1.0)
