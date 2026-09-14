"""
IR-MAD: Iteratively Reweighted Multivariate Alteration Detection.

Canty & Nielsen (2008), "Automatic radiometric normalization of multitemporal
satellite imagery with the iteratively re-weighted MAD transformation",
Remote Sensing of Environment 112(3).

Why this rather than a plain band difference or change-vector analysis
--------------------------------------------------------------------
The hard part of bi-temporal change detection is that two acquisitions of the
same unchanged ground almost never have the same pixel values. Sun angle,
atmospheric path, sensor gain drift and processing baseline all apply a broadly
affine transform to the radiometry. A difference or CVA magnitude cannot tell
that apart from real change: brighten every pixel in T2 by 10% and CVA reports
change everywhere.

MAD is built out of canonical correlation analysis between the two images, and
CCA is *invariant to affine transformation of either image*. So a uniform gain
or offset between dates produces no MAD signal at all — only the departures
from the shared linear relationship survive, which is exactly what change is.

Two further properties matter in practice:

* The MAD variates are mutually orthogonal and individually scaled by their own
  variance, so the sum of squared standardised variates is chi-squared with p
  degrees of freedom under the no-change hypothesis. That gives a *calibrated*
  threshold — "pixels whose no-change probability is below 1e-4" — rather than
  Otsu on an arbitrary magnitude whose scale changes with every scene pair.
* The iterative reweighting progressively down-weights changed pixels when
  estimating the covariance structure, so the no-change model is fitted on the
  no-change population instead of being dragged by the very pixels it is meant
  to detect.

This module is pure numpy/scipy and has no raster dependencies, so it can be
exercised directly on arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cholesky, solve_triangular, svd
from scipy.stats import chi2


@dataclass
class IRMADResult:
    """Outcome of an IR-MAD run over a co-registered image pair."""

    #: Chi-squared statistic per pixel, NaN where invalid. High = likely change.
    chi_squared: np.ndarray
    #: Pr(no change) per pixel under the chi-squared null, NaN where invalid.
    no_change_probability: np.ndarray
    #: Standardised MAD variates, shape (bands, height, width).
    mad_variates: np.ndarray
    #: Canonical correlations, ascending — MAD_1 corresponds to rho[0].
    canonical_correlations: np.ndarray
    #: Iterations actually run before convergence.
    iterations: int
    #: True when the correlations stopped moving within tolerance.
    converged: bool
    #: Factor the raw statistic was divided by to match the chi-squared null.
    #: Far from 1 means the reweighting sharpened heavily, or that the
    #: majority-unchanged assumption is shaky.
    calibration_scale: float = 1.0


def _weighted_moments(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted means and covariance blocks of the stacked [X; Y] vector."""
    total = weights.sum()
    if total <= 0:
        raise np.linalg.LinAlgError("all pixels carry zero weight")

    mean_x = (x * weights).sum(axis=1) / total
    mean_y = (y * weights).sum(axis=1) / total

    centred_x = x - mean_x[:, None]
    centred_y = y - mean_y[:, None]

    weighted_x = centred_x * weights
    # Unbiased-ish scaling; the constant cancels out of the eigenproblem.
    denominator = total - 1.0 if total > 1.0 else total

    sigma_xx = (weighted_x @ centred_x.T) / denominator
    sigma_yy = ((centred_y * weights) @ centred_y.T) / denominator
    sigma_xy = (weighted_x @ centred_y.T) / denominator

    # Ridge keeps Cholesky viable on flat or highly correlated bands.
    bands = x.shape[0]
    scale = max(float(np.trace(sigma_xx)), float(np.trace(sigma_yy)), 1.0) / bands
    eye = np.eye(bands) * (ridge * scale)

    return sigma_xx + eye, sigma_yy + eye, sigma_xy, mean_x, mean_y


def _canonical_vectors(
    sigma_xx: np.ndarray,
    sigma_yy: np.ndarray,
    sigma_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Canonical correlation analysis by whitening plus SVD.

    Returns (A, B, rho) with columns scaled so that AᵀΣxxA = BᵀΣyyB = I, and
    rho ascending so that column 0 is the *least* correlated pair — that is the
    MAD variate carrying the most change signal.
    """
    lower_x = cholesky(sigma_xx, lower=True)
    lower_y = cholesky(sigma_yy, lower=True)

    # K = Lx⁻¹ Σxy Ly⁻ᵀ, solved rather than inverted for conditioning.
    temp = solve_triangular(lower_x, sigma_xy, lower=True)
    kernel = solve_triangular(lower_y, temp.T, lower=True).T

    u, rho, vt = svd(kernel)

    a = solve_triangular(lower_x.T, u, lower=False)
    b = solve_triangular(lower_y.T, vt.T, lower=False)

    # Fix the sign so each canonical pair is positively correlated; otherwise
    # the difference aᵀX − bᵀY can be dominated by an arbitrary sign flip.
    for i in range(b.shape[1]):
        if float(a[:, i] @ sigma_xy @ b[:, i]) < 0:
            b[:, i] = -b[:, i]

    # SVD gives descending correlations; MAD numbering runs the other way.
    order = np.argsort(rho)
    return a[:, order], b[:, order], rho[order]


def irmad(
    first: np.ndarray,
    second: np.ndarray,
    valid: Optional[np.ndarray] = None,
    max_iterations: int = 25,
    tolerance: float = 1e-5,
    ridge: float = 1e-6,
) -> IRMADResult:
    """
    Run IR-MAD over a co-registered pair.

    Parameters
    ----------
    first, second:
        Arrays shaped (bands, height, width). Must already be on a common grid;
        MAD assumes pixel i in one is pixel i in the other, and misregistration
        shows up as change along every edge in the scene.
    valid:
        Boolean (height, width) mask of pixels usable in both images. Invalid
        pixels are excluded from the statistics and returned as NaN.
    max_iterations:
        Ceiling on the reweighting loop.
    tolerance:
        Convergence threshold on the largest change in canonical correlation.
    ridge:
        Relative ridge added to the covariance diagonals for conditioning.

    Raises
    ------
    numpy.linalg.LinAlgError
        When the covariance structure is degenerate — a single band, a constant
        image, or too few valid pixels. Callers should fall back to a simpler
        magnitude in that case.
    """
    if first.ndim != 3 or second.ndim != 3:
        raise ValueError("expected (bands, height, width) arrays")
    if first.shape != second.shape:
        raise ValueError(f"shape mismatch: {first.shape} vs {second.shape}")

    bands, height, width = first.shape
    if bands < 2:
        raise np.linalg.LinAlgError("IR-MAD needs at least two bands")

    x_all = first.reshape(bands, -1).astype(np.float64, copy=False)
    y_all = second.reshape(bands, -1).astype(np.float64, copy=False)

    if valid is None:
        valid_flat = np.ones(height * width, dtype=bool)
    else:
        valid_flat = valid.reshape(-1).astype(bool)

    valid_flat = valid_flat & np.isfinite(x_all).all(axis=0) & np.isfinite(y_all).all(axis=0)

    # 20 samples per covariance entry is a loose but useful floor.
    if int(valid_flat.sum()) < max(64, 20 * bands * bands):
        raise np.linalg.LinAlgError("too few valid pixels for a stable covariance estimate")

    x = x_all[:, valid_flat]
    y = y_all[:, valid_flat]

    weights = np.ones(x.shape[1], dtype=np.float64)
    previous_rho = np.zeros(bands)
    converged = False
    iterations = 0
    mad = np.zeros_like(x)
    rho = previous_rho

    for iterations in range(1, max_iterations + 1):
        sigma_xx, sigma_yy, sigma_xy, mean_x, mean_y = _weighted_moments(
            x, y, weights, ridge
        )
        a, b, rho = _canonical_vectors(sigma_xx, sigma_yy, sigma_xy)

        # MAD_i = aᵢᵀ(X − μx) − bᵢᵀ(Y − μy).
        #
        # The centring is not cosmetic. The canonical vectors come from
        # covariances, which are already centred, so projecting raw radiances
        # leaves a constant offset aᵢᵀμx − bᵢᵀμy in every variate. On a pair
        # with a radiometric shift between dates that offset dwarfs σᵢ, the
        # chi-squared statistic saturates, every pixel is called change, and
        # the reweighting collapses to all-zero on the next pass. Centring is
        # what makes the transform invariant to the offset in the first place.
        mad = (a.T @ (x - mean_x[:, None])) - (b.T @ (y - mean_y[:, None]))

        # Standardise by the no-change deviation σᵢ = sqrt(2(1 − ρᵢ)), which
        # makes the sum of squares chi-squared with p degrees of freedom.
        sigma = np.sqrt(np.maximum(2.0 * (1.0 - rho), 1e-12))
        mad /= sigma[:, None]

        statistic = (mad ** 2).sum(axis=0)

        # Reweight by the probability that this pixel is unchanged, so the next
        # covariance estimate is dominated by the no-change population.
        next_weights = chi2.sf(statistic, bands)

        if not np.isfinite(next_weights).any() or next_weights.sum() <= 1e-12:
            # Every pixel read as change. On a well-correlated pair that means
            # the statistic has saturated numerically, not that the whole scene
            # changed, so keep the last usable fit rather than diverging.
            break

        weights = next_weights
        shift = float(np.max(np.abs(rho - previous_rho)))
        previous_rho = rho
        if shift < tolerance:
            converged = True
            break

    statistic = (mad ** 2).sum(axis=0)

    # ------------------------------------------------------------------
    # Calibrate the statistic against the chi-squared null.
    #
    # The reweighting deliberately fits the covariance to the *most* unchanged
    # pixels, so σᵢ ends up smaller than the spread across the whole scene and
    # every statistic is inflated by a constant factor — on a synthetic
    # no-change pair the median came out at 11.6 against a chi²₄ median of
    # 3.36. Left alone, "p < 1e-4" would flag an order of magnitude more pixels
    # than it claims.
    #
    # Rescaling by the observed median restores the meaning of the p-value.
    # This assumes most of the scene is unchanged, which is the same assumption
    # every unsupervised change detector rests on; it degrades gracefully, and
    # `calibration_scale` is reported so a caller can see how far off it was.
    calibration_scale = 1.0
    observed_median = float(np.median(statistic)) if statistic.size else 0.0
    expected_median = float(chi2.median(bands))
    if np.isfinite(observed_median) and observed_median > 0:
        calibration_scale = observed_median / expected_median
        if calibration_scale > 0:
            statistic = statistic / calibration_scale

    chi_squared = np.full(height * width, np.nan, dtype=np.float32)
    chi_squared[valid_flat] = statistic.astype(np.float32)

    probability = np.full(height * width, np.nan, dtype=np.float32)
    probability[valid_flat] = chi2.sf(statistic, bands).astype(np.float32)

    variates = np.full((bands, height * width), np.nan, dtype=np.float32)
    variates[:, valid_flat] = mad.astype(np.float32)

    return IRMADResult(
        calibration_scale=calibration_scale,
        chi_squared=chi_squared.reshape(height, width),
        no_change_probability=probability.reshape(height, width),
        mad_variates=variates.reshape(bands, height, width),
        canonical_correlations=rho,
        iterations=iterations,
        converged=converged,
    )


def change_magnitude(result: IRMADResult) -> np.ndarray:
    """
    Chi-squared statistic as a change magnitude, NaN outside the valid mask.

    Kept separate from the threshold so the existing pipeline can still run its
    own Otsu/percentile logic over the same surface if it wants to.
    """
    return result.chi_squared


def change_mask(
    result: IRMADResult,
    significance: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    """
    Threshold at a fixed no-change probability.

    Unlike Otsu, this threshold means something the same way on every scene
    pair: `significance` is the per-pixel false-positive rate under the
    no-change null. 1e-4 over a 10⁷-pixel scene still admits ~1000 false
    positives, which is why the caller should follow this with the existing
    morphological cleaning and minimum-area filter.

    Returns (boolean mask, chi-squared threshold used).
    """
    degrees = int(result.mad_variates.shape[0])
    threshold = float(chi2.isf(significance, degrees))
    mask = np.zeros(result.chi_squared.shape, dtype=bool)
    finite = np.isfinite(result.chi_squared)
    mask[finite] = result.chi_squared[finite] > threshold
    return mask, threshold
