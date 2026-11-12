"""Line-of-sight integration into temperature and E-mode transfer functions.

Instead of following the photon hierarchy to ``l ~ 2500``, one integral of the
sources (:mod:`discocmb.sources`) against spherical Bessel functions gives the
temperature transfer function,

    ``Theta_l(k) = int_0^{tau_0} dtau [S j_l(x) + S_j1 j_l'(x) + S_j2 j_l''(x)]``

with ``x = k (tau_0 - tau)``. The E-mode transfer function integrates the
quadrupole term against ``j_l(x) / x^2``,

    ``E_l(k) = sqrt((l+2)!/(l-2)!) int_0^{tau_0} dtau S_j2 j_l(x) / x^2``

(Zaldarriaga & Seljak 1997, CAMB's ``sources(2)``). The projection tabulates
``j_l(x) / x^2`` directly as a fourth Bessel row. Dividing after linear
interpolation would fail near ``x = 0``, where ``j_2(x) / x^2 -> 1/15`` and a
reionized universe still has visibility.

**Projection.** :func:`projection_operator` and :func:`theta_ell_batch` work
as matrix products on grids the batch shares. A 4-point Lagrange stencil
shifts each cosmology's sources onto one grid ``chi = tau0_ref - tau``, a
precomputed spline matrix refines them in ``ln k``, and a batched GEMM
contracts them against one weighted table of ``j_l``, ``j_l'`` and ``j_l''``.
The shared ``chi`` grid costs 2e-4 in ``D_l`` and ``float32`` another 2.6e-4.
It takes 0.25 ms per cosmology (2.3 ms in ``float64``), against 13 ms for a
per-cosmology spline and Bessel evaluation. FFTLog lost: it needs 32768
log-spaced points to resolve recombination and costs ten times more.
"""

from __future__ import annotations

import functools
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from discoeb.spline_interpolation import spline_interpolation
from scipy import special

from .sources import assemble_sources

CACHE_SIZE = 8
"""Number of grid sets whose projection operators and lensing tables stay cached."""


SHIFT_STENCIL = 4
"""Points of the Lagrange stencil that shifts sources onto the shared ``chi`` grid.

4 (cubic) leaves 2e-4 in ``D_l``, 2 (linear) 7e-3."""

PROJECTION_BUDGET_BYTES = 1.0e9
"""Device memory for one projection chunk's fine-``k`` sources. It sets how many
fine wave modes each chunk contracts."""


def array_cache(function):
    """Wrap ``function`` in a :func:`functools.lru_cache` of :data:`CACHE_SIZE` that accepts NumPy arrays.

    The key uses each array's dtype, shape and contents, and ``function``
    receives read-only copies. Other arguments must be hashable.
    """

    def key(value):
        if isinstance(value, np.ndarray):
            return (np.ndarray, value.dtype.str, value.shape, value.tobytes())
        return value

    def unkey(value):
        if isinstance(value, tuple) and len(value) == 4 and value[0] is np.ndarray:
            _, dtype, shape, data = value
            return np.frombuffer(data, dtype=dtype).reshape(shape)
        return value

    @functools.lru_cache(maxsize=CACHE_SIZE)
    def cached(*keys):
        return function(*(unkey(k) for k in keys))

    @functools.wraps(function)
    def wrapper(*args):
        return cached(*(key(a) for a in args))

    return wrapper


def trapezoid_weights(x, xp=np):
    """Return trapezoid weights ``w`` on ``x``, so that ``int f ~ sum_j w_j f(x_j)``.

    ``xp`` is ``np`` on the host or ``jnp`` under ``jit``. The weights stay
    positive whichever way ``x`` runs.
    """

    half = 0.5 * xp.abs(xp.diff(x))
    return xp.pad(half, (0, 1)) + xp.pad(half, (1, 0))


def primordial_spectrum(k, n_s, A_s, k_pivot):
    """Return ``P_R(k) = A_s (k / k_pivot)^(n_s - 1)`` of shape ``(n_c, *k.shape)``.

    ``n_s`` and ``A_s`` are ``(n_c,)``, and ``k`` can have any shape.
    """

    batch = (slice(None),) + (None,) * jnp.ndim(k)
    return A_s[batch] * (k / k_pivot) ** (n_s[batch] - 1.0)


def _bessel_row(ell: int, n_x: int, dx: float) -> np.ndarray:
    """Return ``j_ell`` sampled on the uniform grid ``x = 0, dx, ..., dx (n_x-1)``.

    ``j_ell(x)`` is exponentially small below its turning point ``x ~ ell``, so
    the row stays zero up to ``ell - 4 ell^{1/3}``.
    """

    x = np.linspace(0.0, dx * (n_x - 1), n_x)
    x_safe = np.where(x > 1.0e-30, x, 1.0)
    pref = np.where(x > 1.0e-30, np.sqrt(np.pi / (2.0 * x_safe)), 0.0)
    values = np.zeros_like(x)
    x_min = max(0.0, float(ell) - 4.0 * float(ell) ** (1.0 / 3.0))
    start = max(0, int(x_min / dx) - 1)
    values[start:] = pref[start:] * special.jv(float(ell) + 0.5, x[start:])
    if ell == 0:
        values[0] = 1.0
    return values


@array_cache
def _bessel_tables(ell_arr, x_max, dx):
    """Return uniform tables of ``j_ell``, ``j_{ell+1}`` and ``j_ell / x^2`` up to ``x_max``."""

    n_x = int(np.ceil(x_max / dx)) + 2
    unique = [int(ell) for ell in np.unique(np.concatenate([ell_arr, ell_arr + 1]))]
    # Each row takes a few hundred thousand calls to scipy's j_nu, which
    # releases the GIL, so compute the rows on every core (85 s on one).
    # j_l / x^2 reuses the j_l row.
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        rows = dict(
            zip(unique, pool.map(lambda ell: _bessel_row(ell, n_x, dx), unique))
        )
    return (
        1.0 / dx,
        n_x,
        jnp.asarray(np.stack([rows[int(ell)] for ell in ell_arr])),
        jnp.asarray(np.stack([rows[int(ell) + 1] for ell in ell_arr])),
        jnp.asarray(
            np.stack([_over_x2(rows[int(ell)], int(ell), n_x, dx) for ell in ell_arr])
        ),
    )


def _over_x2(row: np.ndarray, ell: int, n_x: int, dx: float) -> np.ndarray:
    """Return the E-mode kernel ``j_ell(x) / x^2`` from the ``j_ell`` row.

    It is smooth through ``x = 0`` (``1/15`` for ``ell = 2``, zero above), so
    it interpolates well where dividing ``j_ell`` afterwards would not.
    """

    x = np.linspace(0.0, dx * (n_x - 1), n_x)
    x_safe = np.where(x > 1.0e-30, x, 1.0)
    values = row / x_safe**2
    values[0] = 1.0 / 15.0 if ell == 2 else 0.0
    return values


def _uniform_stencil(x, inv_dx, n_x):
    """Return ``(idx, frac)`` for linear interpolation at ``x`` in a table that starts at ``x = 0``.

    Every table on the same ``x`` can reuse them: ``(1 - frac) vals[idx] + frac vals[idx + 1]``.
    """

    u = x * inv_dx
    idx = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n_x - 2)
    return idx, jnp.clip(u - idx, 0.0, 1.0)


@jax.jit
def k_band(ells, k, chi_max, chi_star):
    """Return the ``(n_ell, n_k)`` mask of wave modes that project onto each multipole.

    ``j_l(x)`` is exponentially small below its turning point, so modes with
    ``k chi_max < l - 4 l^{1/3}`` add nothing. Modes far above ``k ~ l /
    chi_star`` add only fast-oscillating tails that the grids cannot resolve,
    which means aliasing noise. Without the upper cut the damping tail comes
    out several percent high (6 percent at ``l = 500``).

    With a batch of ``chi_star``, ``(n_c,)``, the mask becomes ``(n_c, n_ell,
    n_k)``.
    """

    ells = ells[:, None]
    chi_star = jnp.asarray(chi_star)[..., None, None]
    k_lo = jnp.maximum(0.0, ells - 4.0 * ells ** (1.0 / 3.0)) / chi_max
    k_hi = (ells + 2500.0) / chi_star
    return (k[None, :] >= k_lo) & (k[None, :] <= k_hi)


def lagrange_shift_stencil(tau, delta, order=SHIFT_STENCIL):
    """Return the stencil ``(i0, w)`` that evaluates a ``tau``-sampled function at ``tau + delta``.

    ``i0`` is ``(n_tau,)`` and ``w`` is ``(n_tau, order)``, with
    ``f(tau_j + delta) ~ sum_a w[j, a] f(tau[i0_j + a])``. Targets outside the
    grid clamp to its ends. A ``delta`` of shape ``(n_c,)``, one shift per
    cosmology, adds that leading axis to ``i0`` and ``w``.
    """

    tau = np.asarray(tau, dtype=np.float64)
    n = len(tau)
    target = np.clip(
        np.asarray(delta, dtype=np.float64)[..., None] + tau, tau[0], tau[-1]
    )
    i0 = np.clip(np.searchsorted(tau, target) - order // 2, 0, n - order)
    w = np.ones(target.shape + (order,))
    for a in range(order):
        for b in range(order):
            if a != b:
                w[..., a] *= (target - tau[i0 + b]) / (tau[i0 + a] - tau[i0 + b])
    return i0, w


def shift_onto_chi(x, shift_idx, shift_w):
    """Shift ``x``, ``(n_c, ..., n_tau)``, onto the shared ``chi`` grid along its last axis.

    ``shift_idx`` ``(n_c, n_tau)`` and ``shift_w`` ``(n_c, n_tau, order)`` hold
    each cosmology's stencil from :func:`lagrange_shift_stencil`.
    """

    lead = (slice(None),) + (None,) * (x.ndim - 2)
    return sum(
        shift_w[lead + (slice(None), a)]
        * jnp.take_along_axis(x, (shift_idx + a)[lead + (slice(None),)], axis=-1)
        for a in range(shift_w.shape[-1])
    )


class ProjectionOperator(NamedTuple):
    """Shared factors of the line-of-sight projection, from :func:`projection_operator`."""

    interp: jax.Array
    bessel: jax.Array
    bessel_e: jax.Array
    lnk_weights: jax.Array


def projection_operator(ells, k_values, k_fine, tau_save, tau0_ref, dtype=jnp.float32):
    """Return the shared factors of the line-of-sight projection, cached per grid set.

    ``interp``: the ``(n_k_fine, n_k)`` natural cubic spline matrix in
    ``ln k``, the spline of the identity.

    ``bessel``: the ``(n_k_fine, n_ell, 3, n_chi)`` table of ``j_l``,
    ``j_l'`` and ``j_l''`` at ``x = k chi``, with ``chi = tau0_ref -
    tau_save`` and the trapezoid weights of the ``tau`` integral folded in.
    For the three shifted, refined source channels ``A``,

        ``Theta_l(k) = sum_s sum_j A[s, k, j] bessel[k, l, s, j]``

    ``bessel_e``: the ``(n_k_fine, n_ell, n_chi)`` table of
    ``sqrt((l+2)!/(l-2)!) j_l(x) / x^2`` with the same weights, for the third
    channel only.

    ``lnk_weights``: trapezoid weights of the ``C_l`` quadrature in ``ln k``.
    """

    return _projection_operator(
        np.asarray(ells, dtype=np.int64),
        np.asarray(k_values, dtype=np.float64),
        np.asarray(k_fine, dtype=np.float64),
        np.asarray(tau_save, dtype=np.float64),
        float(tau0_ref),
        dtype,
    )


@array_cache
def _projection_operator(ells, k_values, k_fine, tau_save, tau0_ref, dtype):
    interp = spline_interpolation(
        jnp.log(jnp.asarray(k_values)), jnp.eye(len(k_values))
    ).evaluate(jnp.log(jnp.asarray(k_fine)))

    chi = tau0_ref - tau_save
    x = jnp.asarray(k_fine)[:, None] * jnp.asarray(chi)[None, :]
    # round x_max up so nearby grids share one table
    x_max = float(np.max(k_fine) * np.max(chi)) + 2.0
    inv_dx, n_x, jl_tab, jl1_tab, jlx2_tab = _bessel_tables(
        ells, float(np.ceil(x_max / 250.0) * 250.0), 0.03
    )
    weights = jnp.asarray(trapezoid_weights(tau_save))
    # every multipole and table shares x, so interpolate it once
    idx, frac = _uniform_stencil(x, inv_dx, n_x)
    inv_x = jnp.where(x > 1.0e-30, 1.0 / jnp.where(x > 1.0e-30, x, 1.0), 0.0)

    def interpolate(vals):
        return (1.0 - frac) * vals[idx] + frac * vals[idx + 1]

    def rows(i, ell):
        jl = interpolate(jl_tab[i])
        jl1 = interpolate(jl1_tab[i])
        jld = ell * inv_x * jl - jl1
        jldd = -2.0 * inv_x * jld + (ell * (ell + 1.0) * inv_x * inv_x - 1.0) * jl
        return jnp.stack([jl, jld, jldd]) * weights

    def row_e(i, ell):
        prefactor = math.sqrt((ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0))
        return interpolate(jlx2_tab[i]) * (prefactor * weights)

    # Store as (n_k_fine, n_ell, 3, n_chi), the operand layout of the
    # contraction in theta_ell_batch. With (n_ell, 3, n_k_fine, n_chi) XLA
    # transposes the table, gigabytes at inference grids, on every call.
    bessel = jnp.stack(
        [
            jnp.transpose(rows(i, float(ell)), (1, 0, 2)).astype(dtype)
            for i, ell in enumerate(ells)
        ],
        axis=1,
    )
    bessel_e = jnp.stack(
        [row_e(i, float(ell)).astype(dtype) for i, ell in enumerate(ells)], axis=1
    )

    lnk_weights = trapezoid_weights(np.log(k_fine))
    return ProjectionOperator(
        interp.astype(dtype), bessel, bessel_e, jnp.asarray(lnk_weights)
    )


@functools.partial(jax.jit, static_argnames=("k_chunk",))
def theta_ell_batch(
    channels, tau, shift_idx, shift_w, band, interp, bessel, bessel_e, k_chunk=None
):
    """Return ``(Theta_l, E_l)`` at ``k_fine`` for a batch, each ``(n_c, n_ell, n_k_fine)`` in float64.

    ``channels`` is ``(n_c, 5, n_k, n_tau)`` on the shared save grid.
    ``shift_idx`` and ``shift_w`` hold each cosmology's stencil from
    :func:`lagrange_shift_stencil`, and ``band`` is the ``(n_c, n_ell,
    n_k_fine)`` mask of :func:`k_band`. ``interp``, ``bessel`` and
    ``bessel_e`` come from :func:`projection_operator`.

    The contraction handles ``k_chunk`` fine modes at a time (all by default)
    for the whole batch. Each call then reads the Bessel table, gigabytes at
    production sizes, only once, and holds only one chunk of refined sources.
    Chunking does not change the result.
    """

    dtype = bessel.dtype
    sources = assemble_sources(channels, tau[None, None, :])  # (c, 3, k, tau)
    shifted = shift_onto_chi(sources, shift_idx, shift_w).astype(dtype)
    n_fine = interp.shape[0]
    step = n_fine if k_chunk is None else max(1, min(int(k_chunk), n_fine))
    thetas, thetas_e = [], []
    for f0 in range(0, n_fine, step):
        f1 = min(f0 + step, n_fine)
        # put k_fine first as the batch axis of the Bessel contraction. With
        # (c, 3, k_fine, chi) XLA transposed the refined array first, and
        # (k_fine, l, c) out is the batched GEMM's own layout
        refined = jnp.einsum("fk,cskt->fcst", interp[f0:f1], shifted)
        thetas.append(jnp.einsum("fcst,flst->flc", refined, bessel[f0:f1]))
        thetas_e.append(jnp.einsum("fct,flt->flc", refined[:, :, 2], bessel_e[f0:f1]))
    theta = jnp.transpose(jnp.concatenate(thetas), (2, 1, 0))  # (c, l, k_fine)
    theta_e = jnp.transpose(jnp.concatenate(thetas_e), (2, 1, 0))
    return (
        jnp.where(band, theta, 0.0).astype(jnp.float64),
        jnp.where(band, theta_e, 0.0).astype(jnp.float64),
    )


def projection_k_chunk(n_cosmologies, n_k_fine, n_tau, dtype) -> int:
    """Return how many fine modes per :func:`theta_ell_batch` chunk fit :data:`PROJECTION_BUDGET_BYTES`."""

    per_mode = 3 * n_cosmologies * n_tau * jnp.dtype(dtype).itemsize * 2
    return max(1, min(n_k_fine, int(PROJECTION_BUDGET_BYTES // max(per_mode, 1))))


class TransferBatch(NamedTuple):
    """Cosmology-dependent transfer tables of a batch, from :func:`~discocmb.main.compute_transfer_batch`.

    The spectra are quadratic forms of these tables weighted by ``P_R(k)``
    (:func:`~discocmb.spectra.spectra_from_transfer`), so changing only
    ``A_s`` or ``n_s`` needs no new solve.
    """

    theta: jax.Array  # (n_c, n_ell, n_k_fine) temperature transfer Theta_l(k), float64, zero outside each l's k band
    theta_e: jax.Array  # (n_c, n_ell, n_k_fine) E-mode transfer
    theta_psi: (
        jax.Array | None
    )  # (n_c, n_ell_psi_exact, n_k) lensing-potential transfer at the solved modes
    limber: (
        jax.Array | None
    )  # (n_c, n_ell_psi_limber, n_chi) the Limber integrand without P_R
    T_cmb: jax.Array  # (n_c,)


class TransferGrids(NamedTuple):
    """Cosmology-independent arrays that :func:`~discocmb.spectra.spectra_from_transfer` contracts a :class:`TransferBatch` with."""

    ells: np.ndarray
    k_fine: jax.Array
    lnk_weights: jax.Array
    k_pivot: float
    ells_psi: np.ndarray | None  # the lensing-potential multipoles, exact ones first
    k_values: jax.Array | None  # the solved modes, ascending
    k_limber: jax.Array | None  # (n_ell_limber, n_chi) k_chi of the Limber multipoles
