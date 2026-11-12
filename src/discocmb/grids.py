"""Grids for a CMB computation, shaped by a fiducial cosmology.

The sources vary most at recombination and on the sound-horizon scale in
``tau``, and at the acoustic and Silk damping scales in ``k``.
:func:`cmb_tau_grid` and :func:`cmb_k_grid` put their points there. One
trajectory of the source kernel supplies these scales
(:func:`_kernel_thermal_history`), so building the grids compiles nothing
new. :func:`cmb_grids` bundles them into :class:`CMBGrids`, which every batch
of an inference loop can share.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import discoeb as eb
import numpy as np
from discoeb import TAU_START
from scipy import integrate

from .sources import solve_sources, visibility_on_grid

SUMMARY_GRID = 2048
"""Points of the uniform ``log tau`` grid that the grid summary samples the opacity on."""

DEFAULT_CMB_ELLS = np.unique(
    np.concatenate(
        [np.arange(2, 40, 2), np.arange(40, 200, 5), np.arange(200, 2501, 50)]
    )
).astype(np.int64)
"""Default multipoles for ``C_l``."""


Z_REION_SEARCH_MAX = 30.0
"""Redshift below which to look for a reionization bump in the visibility."""


TAU_END_MARGIN = 0.01
"""Fraction past the fiducial ``tau0`` where the save grid of :func:`cmb_grids` ends.

Any cosmology with a conformal age inside the margin can use the fiducial
grids. Against per-batch grids on the tests' 128 cosmologies, the shift moves
``D_l`` by at most 2.7e-4 at 1 percent, 1.0e-3 at 2 percent and 4.4e-3 at 5
percent."""


def _grhoa4(a, grhog, grhor, grhoc, grhob, grhov):
    """Return ``8 pi G rho a^4``, so that ``a' = sqrt(grhoa4 / 3)``."""

    return grhog + grhor + (grhoc + grhob) * a + grhov * a**4


def _dtau_da(a, *dens):
    """Return ``dtau/da = 1 / a'``."""

    return np.sqrt(3.0 / _grhoa4(a, *dens))


SUMMARY_K = 1.0e-3
"""Wave mode (Mpc^-1) of the single trajectory the grid summary reads the opacity from."""


@functools.lru_cache(maxsize=64)
def _kernel_thermal_history(cosmology, n_grid: int = SUMMARY_GRID, n_save: int = 1000):
    """Return ``(tau_grid, a, opacity, tau0)`` on ``n_grid`` uniform ``log tau`` points.

    One trajectory of :func:`~discocmb.sources.solve_sources` records the
    opacity ``kappa'``, reionization included. It reuses the spectra's kernel,
    so nothing new compiles. The scale factor comes from the background
    quadrature. The kernel's save grid has ``n_save`` times and always starts
    at ``TAU_START``, so the loop reads ``n_save - 1`` points per launch.
    """

    a_bg, tau_bg = eb.background_history(cosmology)
    tau0 = float(tau_bg[-1])
    tau_grid = np.exp(np.linspace(np.log(TAU_START), np.log(tau0), n_grid))
    opacity = np.empty(n_grid)
    for c0 in range(0, n_grid, n_save - 1):
        chunk = tau_grid[c0 : c0 + n_save - 1]
        times = np.full(n_save, tau_grid[-1])
        times[0] = TAU_START
        times[1 : 1 + len(chunk)] = chunk
        # nudge the repeated times past the chunk so they strictly increase
        times[1 + len(chunk) :] += np.arange(1, n_save - len(chunk)) * 1.0e-6
        solution = solve_sources((cosmology,), np.asarray([SUMMARY_K]), times)
        channels = np.asarray(solution.values[0, :, 0, :])  # (6, n_save)
        opacity[c0 : c0 + len(chunk)] = channels[4, 1 : 1 + len(chunk)]
    return tau_grid, np.interp(tau_grid, tau_bg, a_bg), opacity, tau0


def _cmb_grid_summary(cosmology) -> dict[str, float]:
    return _grid_summary(eb.as_cosmology(cosmology))


@functools.lru_cache(maxsize=4096)
def _grid_summary(cosmology) -> dict[str, float]:
    """Return the scales that shape the ``k`` and ``tau`` grids.

    ``tau_star`` is the visibility peak and ``delta_tau_rec`` its Gaussian
    width. ``r_s``, the sound horizon at recombination, sets the acoustic
    period in ``k``, and the Silk damping wavenumber ``k_D`` sets the
    envelope.
    """

    dens = eb.density_coefficients(cosmology)
    tau_grid, a_vals, opacity, tau0 = _kernel_thermal_history(cosmology)

    _, vis, _ = visibility_on_grid(tau_grid, opacity, tau_grid, float(tau0))
    peak = int(np.argmax(vis))
    tau_star = float(tau_grid[peak])
    a_star = float(a_vals[peak])

    half = 0.5 * float(vis[peak])
    left = float(np.interp(half, vis[: peak + 1], tau_grid[: peak + 1]))
    right = float(np.interp(half, vis[peak:][::-1], tau_grid[peak:][::-1]))
    delta_tau_rec = max((right - left) / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1.0)

    grhog_v, grhor_v, grhoc_v, grhob_v, _ = dens
    tau_eq = float(
        np.interp((grhog_v + grhor_v) / (grhoc_v + grhob_v), a_vals, tau_grid)
    )

    # Comoving sound horizon r_s = int c_s dtau = int_0^{a_star} da / (a' sqrt(3(1+R))),
    # with a' = sqrt(grhoa4/3) and R = 3 rho_b / (4 rho_g).
    def sound_integrand(ap):
        r = 0.75 * grhob_v * ap / grhog_v
        return 1.0 / np.sqrt(_grhoa4(ap, *dens) * (1.0 + r))

    r_s = float(integrate.quad(sound_integrand, 0.0, a_star, limit=200)[0])

    # Silk damping scale from the photon diffusion length.
    a_silk = np.linspace(max(float(a_vals[0]), 1.0e-8), a_star, 5000)
    opacity_silk = np.maximum(np.interp(a_silk, a_vals, opacity), 1.0e-30)
    r = 0.75 * grhob_v * a_silk / grhog_v
    integrand_d = (
        (r**2 + 16.0 * (1.0 + r) / 15.0)
        / (6.0 * (1.0 + r) ** 2 * opacity_silk)
        * _dtau_da(a_silk, *dens)
    )
    k_D = float(1.0 / np.sqrt(np.trapezoid(integrand_d, a_silk)))

    # Find the reionization bump in the visibility when tau_reio > 0. Without
    # one, fall back to a broad late-time bump for the save grid's weight.
    post_rec = a_vals > 1.0 / (1.0 + Z_REION_SEARCH_MAX)
    if np.any(post_rec) and np.max(np.where(post_rec, vis, 0.0)) > 1.0e-3 * vis[peak]:
        reion_idx = int(np.argmax(np.where(post_rec, vis, 0.0)))
        tau_reion = float(tau_grid[reion_idx])
        delta_tau_reion = max(0.03 * tau0, 80.0)
    else:
        tau_reion = 0.6 * tau0
        delta_tau_reion = 0.05 * tau0

    out = {
        "tau0": float(tau0),
        "tau_eq": tau_eq,
        "tau_star": tau_star,
        "delta_tau_rec": delta_tau_rec,
        "tau_reion": tau_reion,
        "delta_tau_reion": delta_tau_reion,
        "r_s": r_s,
        "k_D": k_D,
    }
    return out


def _resample_from_density(x, density, n):
    """Return ``n`` points drawn from ``x`` in proportion to ``density``."""

    cdf = np.cumsum(density) * (x[1] - x[0])
    cdf -= cdf[0]
    cdf /= cdf[-1]
    return np.interp(np.linspace(0.0, 1.0, n), cdf, x)


def cmb_k_grid(
    cosmology,
    *,
    n: int = 200,
    mode: str = "ode",
    k_min: float = 1.0e-5,
    k_max: float = 0.5,
    ell_min: int = 2,
    ell_max: int = 2500,
    n_ell_samples: int = 30,
    n_eval: int = 5000,
) -> np.ndarray:
    """Return a nonuniform wavenumber grid, densest where the sources vary.

    ``mode="ode"`` resolves the sources' acoustic oscillations (period from
    ``r_s``, envelope from ``k_D``). The perturbation solve runs on it.
    ``mode="cl"`` also weights each ``k`` by how many multipoles project onto
    it, ``k ~ l / chi_star``. The ``C_l`` quadrature uses this finer grid.
    """

    cosmology = eb.as_cosmology(cosmology)
    summary = _cmb_grid_summary(cosmology)
    x = np.linspace(np.log(k_min), np.log(k_max), n_eval)
    k = np.exp(x)
    primordial = k ** (cosmology.n_s + 2.0)
    acoustic_curv = (1.0 / summary["r_s"]) ** 2
    damped = primordial * np.exp(-((k / summary["k_D"]) ** 2))

    if mode == "cl":
        sigma_k = 1.0 / summary["delta_tau_rec"]
        chi_star = summary["tau0"] - summary["tau_star"]
        ell_samples = np.unique(
            np.geomspace(ell_min, ell_max, n_ell_samples).astype(int)
        )
        raw_weight = np.zeros_like(k)
        for ell in ell_samples:
            envelope = np.exp(-0.5 * ((k - ell / chi_star) / (3.0 * sigma_k)) ** 2)
            raw_weight += np.maximum(acoustic_curv, sigma_k**2) * envelope * damped
        floor = 1.0e-6 * np.max(raw_weight)
    else:
        smooth_curv = (k * summary["r_s"]) ** 2 / summary["tau_eq"] ** 2
        raw_weight = np.maximum(acoustic_curv, smooth_curv) * damped
        floor = 0.005 * np.max(raw_weight)

    grid = np.exp(_resample_from_density(x, (raw_weight + floor) ** (1.0 / 3.0), n))
    grid[0], grid[-1] = k_min, k_max
    return grid


def cmb_tau_grid(
    cosmology, k_values, n: int, *, tau_end: float | None = None, n_eval: int = 10000
) -> np.ndarray:
    """Return a nonuniform conformal-time grid for saving the sources.

    The sources peak sharply at recombination (width ``delta_tau_rec``), vary
    on the sound-horizon scale and may bump at reionization. The sampling
    density is the cube root of the sum of those three weights. The grid
    starts at ``TAU_START`` and ends at ``tau_end``, as the solver requires.

    ``tau_end`` defaults to this cosmology's ``tau0``. A batch passes its
    largest ``tau0`` so no trajectory stops early. ``cosmology`` still sets the
    weights.
    """

    k_arr = np.asarray(k_values, dtype=np.float64)
    s = _cmb_grid_summary(cosmology)
    tau0 = s["tau0"] if tau_end is None else float(tau_end)

    tau = np.linspace(TAU_START, tau0, n_eval, dtype=np.float64)
    weight = (
        np.exp(-0.5 * ((tau - s["tau_star"]) / s["delta_tau_rec"]) ** 2)
        / s["delta_tau_rec"] ** 2
    )
    weight += (
        np.exp(-0.5 * ((tau - s["tau_star"]) / s["r_s"]) ** 2)
        * (float(k_arr[-1]) / np.sqrt(3.0)) ** 2
    )
    weight += (
        0.3
        * np.exp(-0.5 * ((tau - s["tau_reion"]) / s["delta_tau_reion"]) ** 2)
        / s["delta_tau_reion"] ** 2
    )

    tau_save = _resample_from_density(
        tau, (weight + 0.005 * np.max(weight)) ** (1.0 / 3.0), n
    )
    tau_save[0], tau_save[-1] = TAU_START, tau0
    return tau_save


class CMBGrids(NamedTuple):
    """Grids of one CMB computation, shaped by a fiducial cosmology.

    ``k_values`` holds the solved wave modes, ``k_fine`` the quadrature grid
    the sources get refined onto, ``tau_save`` the save times and ``ells`` the
    multipoles. :func:`cmb_grids` builds them. Passing them to
    :func:`~discocmb.main.compute_cl_power_spectrum_batch` keeps one
    projection operator across batches.
    """

    k_values: np.ndarray
    ells: np.ndarray
    k_fine: np.ndarray
    tau_save: np.ndarray

    @property
    def tau_end(self) -> float:
        return float(self.tau_save[-1])


def cmb_grids(
    cosmology,
    *,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    tau_end: float | None = None,
    tau_end_margin: float = TAU_END_MARGIN,
) -> CMBGrids:
    """Return :class:`CMBGrids` shaped by ``cosmology``, to reuse across batches.

    Nearby cosmologies share where their sources vary, so one fiducial's grids
    serve them all. Fixed grids let
    :func:`~discocmb.transfer.projection_operator` build the Bessel table and
    ``ln k`` refinement matrix once instead of spending 0.3 s per batch.

    The save grid ends at ``tau_end``, by default ``tau_end_margin`` past the
    fiducial's ``tau0``. Every cosmology needs ``tau0 <= tau_end``, otherwise
    :func:`~discocmb.main.compute_transfer_batch` raises ``ValueError``. Its
    saves past its own ``tau0`` carry zero sources. A larger margin costs
    accuracy (see :data:`TAU_END_MARGIN`), so keep it as small as the batches
    allow.
    """

    cosmology = eb.as_cosmology(cosmology)
    if k_values is None:
        k_values = cmb_k_grid(cosmology, n=128, mode="ode")
    if ells is None:
        ells = DEFAULT_CMB_ELLS
    k_values = np.asarray(k_values, dtype=np.float64)
    ells = np.asarray(ells, dtype=np.int64)
    if tau_end is None:
        tau_end = _cmb_grid_summary(cosmology)["tau0"] * (1.0 + float(tau_end_margin))
    tau_save = cmb_tau_grid(cosmology, k_values, n_save, tau_end=float(tau_end))
    if k_fine_values is None:
        k_fine = cmb_k_grid(
            cosmology,
            n=n_k_fine,
            mode="cl",
            k_min=float(k_values[0]),
            k_max=float(k_values[-1]),
            ell_max=int(np.max(ells)),
        )
    else:
        k_fine = np.asarray(k_fine_values, dtype=np.float64)
    return CMBGrids(k_values, ells, k_fine, tau_save)
