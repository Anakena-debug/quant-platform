"""Byte-exact regression on ConformalAlphaModel branches.

S14 added method='mondrian' as a fourth opt-in branch. The
split/cv/cqr references in this file were computed at S14
sprint-execute time on the pre-mondrian-add code path (commit
b1ab280..32cfeb3 era HEAD pre-edit) — they pin that adding the
mondrian branch did not silently shift split/cv/cqr outputs.

S15 P14.3 part 1 extends the file with a fourth pin
``test_pin_mondrian_branch_byte_exact``. The mondrian reference
arrays were generated under ``OMP_NUM_THREADS=1`` on commit
``6056f08`` (S15 P14.2 HEAD) with ``numpy 2.4.4`` and
``sklearn 1.8.0``. ``alpha.py`` is forbidden to edit in S15
(Route C invariant), so the mondrian branch is byte-stable
relative to its S14 introduction; the pin locks that.

Stratifier for the mondrian pin: ``(X[:, 0] > 0).astype(int)``
— two strata, deterministic on both the training and test
slices of the fixed-seed synthetic.

A future numpy/sklearn version bump that shifts any reference
fails its pin and forces a conscious re-baseline (with reason
logged in the deviation log).

Same regression-discipline as S12's S8 byte-exact gate, scaled to
ConformalAlphaModel.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from quantcore.uncertainty.conformal.finance.alpha import ConformalAlphaModel


# -----------------------------------------------------------------------------
# Fixed-seed synthetic.
# -----------------------------------------------------------------------------


def _synthetic(seed: int = 42, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = X.sum(axis=1) + 0.3 * rng.standard_normal(n)
    return X, y


# -----------------------------------------------------------------------------
# Pre-S14 reference arrays. Computed on the pre-mondrian-add code
# path with the synthetic above. Each array has 20 elements
# (X_test = X[180:200]).
# -----------------------------------------------------------------------------

REF_SPLIT_EXPECTED = np.array(
    [
        -2.2812481685082964,
        -0.6115052008243999,
        -0.10249070476245878,
        1.5745169437516828,
        2.1808530480464365,
        -2.1016288806045655,
        -2.2162093144930517,
        -0.402387751239088,
        1.4455066662348288,
        0.044746795879371554,
        -1.2844019714020296,
        -2.194529498759632,
        2.2892284588014844,
        -3.174802885086768,
        -0.6568526134262898,
        -0.006201087567441171,
        0.5708573633055838,
        0.3618334508455996,
        -2.2462677594834117,
        -0.7427388064128371,
    ]
)
REF_SPLIT_LOWER = np.array(
    [
        -2.824087338939668,
        -1.1543443712557715,
        -0.6453298751938306,
        1.031677773320311,
        1.6380138776150648,
        -2.644468051035937,
        -2.7590484849244232,
        -0.9452269216704597,
        0.9026674958034571,
        -0.49809237455200017,
        -1.8272411418334014,
        -2.737368669191004,
        1.7463892883701126,
        -3.71764205551814,
        -1.1996917838576615,
        -0.549040257998813,
        0.028018192874212033,
        -0.18100571958577216,
        -2.7891069299147837,
        -1.2855779768442088,
    ]
)
REF_SPLIT_UPPER = np.array(
    [
        -1.7384089980769246,
        -0.0686660303930281,
        0.44034846566891295,
        2.1173561141830546,
        2.723692218477808,
        -1.5587897101731938,
        -1.67337014406168,
        0.14045141919228377,
        1.9883458366662006,
        0.5875859663107433,
        -0.7415628009706579,
        -1.6516903283282602,
        2.8320676292328564,
        -2.631963714655396,
        -0.114013442994918,
        0.5366380828639306,
        1.1136965337369555,
        0.9046726212769713,
        -1.70342858905204,
        -0.19989963598146532,
    ]
)

REF_CV_EXPECTED = np.array(
    [
        -2.291991203964011,
        -0.6044100363557013,
        -0.11357092187493698,
        1.5856831389623032,
        2.2199795433673617,
        -2.092341790533347,
        -2.2226918409934853,
        -0.3826684409157945,
        1.4701242509501349,
        0.05717181946950501,
        -1.2841252022069058,
        -2.170106692364519,
        2.3233849289578568,
        -3.1841060053392063,
        -0.67074606024183,
        0.003905832313968606,
        0.6162991014747464,
        0.41482464922128187,
        -2.241710294427073,
        -0.7077090543810068,
    ]
)
REF_CV_LOWER = np.array(
    [
        -2.820123643402999,
        -1.1325424757946894,
        -0.6417033613139251,
        1.057550699523315,
        1.6918471039283736,
        -2.620474229972335,
        -2.7508242804324734,
        -0.9108008803547827,
        0.9419918115111467,
        -0.47096061996948313,
        -1.812257641645894,
        -2.6982391318035073,
        1.7952524895188686,
        -3.7122384447781944,
        -1.1988784996808182,
        -0.5242266071250196,
        0.08816666203575829,
        -0.11330779021770626,
        -2.769842733866061,
        -1.2358414938199949,
    ]
)
REF_CV_UPPER = np.array(
    [
        -1.7638587645250228,
        -0.07627759691671321,
        0.41456151756405113,
        2.113815578401291,
        2.74811198280635,
        -1.5642093510943589,
        -1.6945594015544971,
        0.14546399852319364,
        1.998256690389123,
        0.5853042589084931,
        -0.7559927627679177,
        -1.641974252925531,
        2.851517368396845,
        -2.655973565900218,
        -0.1426136208028419,
        0.5320382717529567,
        1.1444315409137347,
        0.9429570886602701,
        -1.7135778549880847,
        -0.1795766149420187,
    ]
)

REF_CQR_EXPECTED = np.array(
    [
        -0.6580315509216492,
        -0.17179745165764604,
        -0.25388901949271214,
        0.06762128699266512,
        0.9576052923928846,
        -0.47578033676464127,
        -1.0626510487762997,
        -0.23334763247542523,
        0.2504665237625101,
        -0.046319622850995223,
        -0.5053404933968735,
        -0.5709213817457305,
        0.6343141074878493,
        -0.6309532795711953,
        -0.25388901949271214,
        -0.5945557060255189,
        0.016996651507184657,
        0.8175939598583779,
        -0.641032889685517,
        -0.35881045552597457,
    ]
)
REF_CQR_LOWER = np.array(
    [
        -3.115018711301548,
        -2.0936936588423034,
        -2.4337567219915077,
        -2.0936936588423034,
        -2.0936936588423034,
        -2.701659429056294,
        -3.8754008530796105,
        -2.448712180362665,
        -2.0936936588423034,
        -2.0936936588423034,
        -2.8096365962519965,
        -2.8919415190184723,
        -2.0936936588423034,
        -3.243923474554205,
        -2.4337567219915077,
        -2.8473497536352435,
        -2.616935012297423,
        -2.7546026871893106,
        -3.0321645348980453,
        -2.7546026871893106,
    ]
)
REF_CQR_UPPER = np.array(
    [
        1.7989556094582495,
        1.7500987555270113,
        1.9259786830060834,
        2.2289362328276336,
        4.008904243628073,
        1.7500987555270113,
        1.7500987555270113,
        1.9820169154118144,
        2.5946267063673236,
        2.001054413140313,
        1.7989556094582495,
        1.7500987555270113,
        3.362321873818002,
        1.9820169154118144,
        1.9259786830060834,
        1.6582383415842057,
        2.650928315311792,
        4.389790606906066,
        1.7500987555270113,
        2.0369817761373614,
    ]
)

# Pre-S15 mondrian reference. Generated under OMP_NUM_THREADS=1 on
# commit 6056f08 with numpy 2.4.4 / sklearn 1.8.0, two-strata
# stratifier (sign of feature 0).
REF_MONDRIAN_EXPECTED = np.array(
    [
        -2.2645618014571274,
        -0.6076687499615341,
        -0.18795266077407435,
        1.5701485059059226,
        2.1898859907802595,
        -2.0971689036847376,
        -2.205956943846517,
        -0.4636476309847,
        1.4622493146998097,
        0.06974387911037949,
        -1.2574575880151242,
        -2.156892009744682,
        2.357447693233362,
        -3.318369610411935,
        -0.674886539448792,
        0.03093124626195065,
        0.5450826142398957,
        0.33788112418567606,
        -2.2221435859661347,
        -0.7721967881325138,
    ]
)
REF_MONDRIAN_LOWER = np.array(
    [
        -2.9718032969447314,
        -1.3149102454491381,
        -0.7666483853008451,
        0.9914527813791518,
        1.6111902662534887,
        -2.8044103991723417,
        -2.913198439334121,
        -1.0423433555114707,
        0.8835535901730389,
        -0.6374976163772246,
        -1.9646990835027283,
        -2.864133505232286,
        1.7787519687065911,
        -3.8970653349387057,
        -1.382128034936396,
        -0.5477644782648201,
        -0.03361311028687508,
        -0.2408146003410947,
        -2.9293850814537388,
        -1.3508925126592846,
    ]
)
REF_MONDRIAN_UPPER = np.array(
    [
        -1.5573203059695233,
        0.09957274552606998,
        0.3907430637526964,
        2.1488442304326933,
        2.7685817153070302,
        -1.3899274081971336,
        -1.498715448358913,
        0.11504809354207068,
        2.0409450392265804,
        0.7769853745979836,
        -0.5502160925275201,
        -1.449650514257078,
        2.9361434177601327,
        -2.739673885885164,
        0.03235495603881211,
        0.6096269707887214,
        1.1237783387666664,
        0.9165768487124468,
        -1.5149020904785306,
        -0.19350106360574304,
    ]
)


# -----------------------------------------------------------------------------
# Pins.
# -----------------------------------------------------------------------------


def test_pin_split_branch_byte_exact() -> None:
    """split branch on fixed-seed synthetic produces AlphaSignal
    arrays bitwise-identical to the pre-S14 reference."""
    X, y = _synthetic()
    X_train, y_train = X[:180], y[:180]
    X_test = X[180:]

    m = ConformalAlphaModel(
        model=LinearRegression(),
        alpha=0.1,
        method="split",
        random_state=42,
    )
    m.fit(X_train, y_train)
    sig = m.predict(X_test)

    np.testing.assert_array_equal(sig.expected_return, REF_SPLIT_EXPECTED)
    np.testing.assert_array_equal(sig.lower, REF_SPLIT_LOWER)
    np.testing.assert_array_equal(sig.upper, REF_SPLIT_UPPER)


def test_pin_cv_branch_byte_exact() -> None:
    """cv branch on fixed-seed synthetic produces AlphaSignal
    arrays bitwise-identical to the pre-S14 reference."""
    X, y = _synthetic()
    X_train, y_train = X[:180], y[:180]
    X_test = X[180:]

    m = ConformalAlphaModel(
        model=LinearRegression(),
        alpha=0.1,
        method="cv",
        n_folds=5,
        random_state=42,
    )
    m.fit(X_train, y_train)
    sig = m.predict(X_test)

    np.testing.assert_array_equal(sig.expected_return, REF_CV_EXPECTED)
    np.testing.assert_array_equal(sig.lower, REF_CV_LOWER)
    np.testing.assert_array_equal(sig.upper, REF_CV_UPPER)


def test_pin_cqr_branch_byte_exact() -> None:
    """cqr branch on fixed-seed synthetic produces AlphaSignal
    arrays bitwise-identical to the pre-S14 reference."""
    X, y = _synthetic()
    X_train, y_train = X[:180], y[:180]
    X_test = X[180:]

    gbr = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    m = ConformalAlphaModel(
        model=gbr,
        alpha=0.1,
        method="cqr",
        random_state=42,
    )
    m.fit(X_train, y_train)
    sig = m.predict(X_test)

    np.testing.assert_array_equal(sig.expected_return, REF_CQR_EXPECTED)
    np.testing.assert_array_equal(sig.lower, REF_CQR_LOWER)
    np.testing.assert_array_equal(sig.upper, REF_CQR_UPPER)


def test_pin_mondrian_branch_byte_exact() -> None:
    """mondrian branch on fixed-seed synthetic produces AlphaSignal
    arrays bitwise-identical to the pre-S15 reference (commit
    6056f08, numpy 2.4.4, sklearn 1.8.0).

    Two-strata stratifier on sign of feature 0; deterministic on
    both train and test slices. ``alpha.py`` is forbidden in S15
    (Route C invariant), so this pin locks mondrian's outputs at
    the S14 introduction shape against future drift.
    """
    X, y = _synthetic()
    X_train, y_train = X[:180], y[:180]
    X_test = X[180:]

    def _stratifier(X_in: np.ndarray) -> np.ndarray:
        return (X_in[:, 0] > 0).astype(np.int_)

    m = ConformalAlphaModel(
        model=LinearRegression(),
        alpha=0.1,
        method="mondrian",
        random_state=42,
        stratifier=_stratifier,
        mondrian_base_method="split",
    )
    m.fit(X_train, y_train)
    sig = m.predict(X_test)

    np.testing.assert_array_equal(sig.expected_return, REF_MONDRIAN_EXPECTED)
    np.testing.assert_array_equal(sig.lower, REF_MONDRIAN_LOWER)
    np.testing.assert_array_equal(sig.upper, REF_MONDRIAN_UPPER)
