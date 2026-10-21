import numpy as np
import pandas as pd

from src.rul import (
    CAP,
    FEATURES,
    WINDOW,
    nasa_score,
    normalise,
    pad,
    test_windows as make_test_windows,
    train_windows,
)


def make_frame(unit, cycles):
    cycle = np.arange(1, cycles + 1)
    data = {
        "unit": np.full(cycles, unit),
        "cycle": cycle,
        "set1": np.zeros(cycles),
        "set2": np.zeros(cycles),
        "set3": np.zeros(cycles),
        "rul": np.minimum(cycles - cycle, CAP),
    }
    for index, feature in enumerate(FEATURES, start=1):
        data[feature] = cycle * index
    return pd.DataFrame(data)


def test_pad_repeats_the_first_row_before_short_sequence():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    padded = pad(values)

    assert padded.shape == (WINDOW, 2)
    np.testing.assert_array_equal(
        padded[: WINDOW - len(values)],
        np.repeat(values[:1], WINDOW - len(values), axis=0),
    )
    np.testing.assert_array_equal(padded[-len(values) :], values)


def test_train_windows_covers_each_eligible_cycle_and_caps_targets():
    frame = make_frame(unit=1, cycles=200)

    windows, targets, units, cycles = train_windows(frame)

    assert windows.shape == (200 - WINDOW + 1, WINDOW, len(FEATURES))
    np.testing.assert_array_equal(units, np.ones(200 - WINDOW + 1))
    np.testing.assert_array_equal(cycles, np.arange(WINDOW, 201))
    np.testing.assert_allclose(targets, np.minimum(200 - cycles, CAP))
    assert targets[0] == CAP
    assert targets[-1] == 0


def test_train_windows_returns_one_padded_window_for_a_short_engine():
    frame = make_frame(unit=1, cycles=5)

    windows, targets, units, cycles = train_windows(frame)

    assert windows.shape == (1, WINDOW, len(FEATURES))
    assert targets.tolist() == [0]
    assert units.tolist() == [1]
    assert cycles.tolist() == [5]
    expected = frame[FEATURES].to_numpy(np.float32)
    np.testing.assert_array_equal(
        windows[0, : WINDOW - len(frame)],
        np.repeat(expected[:1], WINDOW - len(frame), axis=0),
    )
    np.testing.assert_array_equal(windows[0, -len(frame) :], expected)


def test_test_windows_returns_one_last_window_per_engine():
    short = make_frame(unit=1, cycles=5)
    long = make_frame(unit=2, cycles=35)

    windows, cycles = make_test_windows(pd.concat([short, long], ignore_index=True))

    assert windows.shape == (2, WINDOW, len(FEATURES))
    np.testing.assert_array_equal(cycles, [5, 35])
    np.testing.assert_array_equal(
        windows[0, : WINDOW - len(short)],
        np.repeat(windows[0, :1], WINDOW - len(short), axis=0),
    )
    np.testing.assert_array_equal(
        windows[0, -len(short) :], short[FEATURES].to_numpy(np.float32)
    )
    np.testing.assert_array_equal(
        windows[1], long[FEATURES].to_numpy(np.float32)[-WINDOW:]
    )


def test_nasa_score_penalises_late_predictions_more_than_early_predictions():
    true = np.array([10.0])

    early = nasa_score(np.array([8.0]), true)
    late = nasa_score(np.array([12.0]), true)

    assert late > early


def test_normalise_fits_statistics_only_on_fit_rows():
    train = make_frame(unit=1, cycles=3)
    test = make_frame(unit=2, cycles=1)
    for feature in FEATURES:
        train[feature] = [1.0, 3.0, 100.0]
        test[feature] = [2.0]
    fit_mask = pd.Series([True, True, False], index=train.index)

    normalise(train, test, n_conditions=1, fit_mask=fit_mask)

    fit_values = train.loc[fit_mask, FEATURES]
    expected_validation = (100.0 - 2.0) / np.sqrt(2.0)
    np.testing.assert_allclose(fit_values.mean(axis=0), 0.0)
    np.testing.assert_allclose(fit_values.std(axis=0), 1.0)
    np.testing.assert_allclose(train.loc[~fit_mask, FEATURES], expected_validation)
    np.testing.assert_allclose(test[FEATURES], 0.0)
