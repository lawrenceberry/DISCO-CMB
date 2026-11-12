"""Weak lensing of the CMB temperature and polarization spectra.

Two parts, both batched over cosmologies:

1. **Lensing potential spectrum** ``C_l^{psi psi}``, from the Weyl potential
   the kernel saves (the fourth source channel of :mod:`discocmb.sources`).
   The lensing potential integrates the Weyl potential along the line of
   sight (Lewis & Challinor 2006, eq. 4.6, CAMB's ``sources(3)``):

   ``psi(n) = -2 int_0^{chi_*} dchi (chi_* - chi) / (chi chi_*) phi_W(chi n, tau_0 - chi)``,

   so ``Theta_l^psi(k) = -2 int dchi W(chi) phi_W(k, tau_0 - chi) j_l(k chi)``
   and ``C_l^{psi psi} = 4 pi int dln k P_R(k) |Theta_l^psi(k)|^2``, with the
   normalization of :func:`discocmb.spectra.cl_from_theta`. The kernel is
   broad in ``chi``, so the transfer function is smooth in ``k``. The solved
   modes suffice for the quadrature, and the projection is a small matrix
   product per cosmology (:func:`lensing_potential_operator`,
   :func:`lensing_potential_transfer`).

2. **Lensed spectra**, by the full-sky correlation-function method of
   Challinor & Lewis (2005, Phys. Rev. D 71, 103010) in the form of CAMB's
   ``CorrFuncFullSky``. It keeps the isotropic part of the deflection to all
   orders, ``F_l = exp(-l(l+1) sigma^2/2)``, and the anisotropic part to
   second order in ``c_l = l(l+1) C_gl,2 / 2``. With ``b_l = (2l+1)/(4pi)``
   the lensed correlation functions are

   ``xi   = sum_l b_l C_l^TT F_l [d_00 + c_l d_{1-1} + c_l^2/4 (d_00 + d_{2-2})]``,
   ``xi_+ = sum_l b_l C_l^EE F_l [d_22 + c_l d_13 + c_l^2/4 (d_22 + d_04)]``,
   ``xi_- = sum_l b_l C_l^EE F_l [d_{2-2} + c_l/2 (d_{1-1} + d_{3-3}) + c_l^2/8 (2 d_{2-2} + d_00 + d_{4-4})]``,
   ``xi_X = sum_l b_l C_l^TE F_l [d_20 + c_l/2 (d_11 + d_{1-3}) + c_l^2/8 (3 d_20 + d_{2-4})]``,

   where ``d_{mm'} = d^l_{mm'}(beta)`` are Wigner d-functions,
   ``sigma^2(beta) = C_gl(0) - C_gl(beta)``, ``C_gl = sum_l b_l l(l+1) C_l^psi
   d^l_11`` and ``C_gl,2`` is the same sum with ``d^l_{1-1}``. Gauss-Legendre
   quadrature in ``cos(beta)`` transforms them back: ``C~_l^TT = 2 pi int xi
   d^l_00``, ``C~_l^TE = 2 pi int xi_X d^l_20``, ``C~_l^EE +- C~_l^BB = 2 pi
   int xi_+- d^l_{2 +-2}``. The d-functions depend only on ``l_max`` and the
   node count, so :func:`lensing_tables` builds them once.
   :func:`lensed_spectra` then needs a few dozen matrix products per
   cosmology. It sums over every multipole, and :func:`cl_on_every_ell`
   splines the sampled spectra onto them.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

from .transfer import (
    array_cache,
    primordial_spectrum,
    shift_onto_chi,
    trapezoid_weights,
)

# ---------------------------------------------------------------------------
# 1. Lensing potential
# ---------------------------------------------------------------------------


def lensing_potential_operator(ells, k_values, tau_save, tau0_ref, dtype=jnp.float32):
    """Return the ``(n_ell, n_k, n_chi)`` table of ``j_l(k chi)`` times the ``chi`` trapezoid weights.

    ``chi = tau0_ref - tau_save`` is the shared comoving-distance grid.
    :func:`lensing_potential_transfer` applies each cosmology's lensing
    kernel. SciPy builds the table on the host in about a second for 60
    multipoles and 128 modes.
    """

    ells = np.asarray(ells, dtype=np.int64)
    k_values = np.asarray(k_values, dtype=np.float64)
    tau_save = np.asarray(tau_save, dtype=np.float64)
    chi = tau0_ref - tau_save
    weights = trapezoid_weights(chi)
    x = k_values[:, None] * chi[None, :]
    table = np.stack([special.spherical_jn(int(ell), x) * weights for ell in ells])
    return jnp.asarray(table, dtype=dtype)


LIMBER_LMIN = 200
"""First multipole of the lensing potential to use the Limber approximation.

The exact integral needs the save grid to resolve ``j_l(k chi)`` along the
whole line of sight, but the grid is coarse at late times. The Limber
integrand is smooth in ``chi``. Against CLASS the exact projection agrees to
0.2 percent below ``l ~ 300`` and Limber to 0.3 percent above ``l ~ 200``.
"""


def limber_wavenumbers(ells, chi):
    """Return ``k_chi = (l + 1/2) / chi``, ``(n_ell, n_chi)``, the Limber wave number at ``chi``.

    Points with ``chi <= 0`` use a distance of 1. They carry no weight.
    """

    chi = np.asarray(chi, dtype=np.float64)
    chi_safe = np.where(chi > 0.0, chi, 1.0)
    return (np.asarray(ells, dtype=np.float64)[:, None] + 0.5) / chi_safe[None, :]


def limber_interpolation(ells, k_values, chi):
    """Return ``(n_ell, n_chi, n_k)`` cubic spline weights in ``ln k`` from the modes to ``k = (l + 1/2) / chi``.

    Targets outside the solved modes get zero weight.
    """

    from discoeb.spline_interpolation import spline_interpolation

    k_values = np.asarray(k_values, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)
    target = limber_wavenumbers(ells, chi)
    inside = (chi > 0.0)[None, :] & (target >= k_values[0]) & (target <= k_values[-1])
    lnk_t = jnp.log(jnp.asarray(np.where(inside, target, k_values[0])))
    weights = spline_interpolation(
        jnp.log(jnp.asarray(k_values)), jnp.eye(len(k_values))
    ).evaluate(lnk_t.ravel())
    weights = (
        weights.reshape(len(ells), len(chi), len(k_values))
        * jnp.asarray(inside)[..., None]
    )
    return weights.astype(jnp.float32)


def _lensing_kernel(chi, chi_star):
    chi_s = chi_star[:, None]
    return jnp.where(
        (chi > 0.0) & (chi < chi_s),
        (chi_s - chi) / jnp.where(chi > 0.0, chi, 1.0) / chi_s,
        0.0,
    )


@jax.jit
def lensing_potential_transfer(
    phi,
    shift_idx,
    shift_w,
    chi,
    chi_star,
    operator,
    limber_weights,
    k_chi,
):
    """Return ``(theta_psi, limber)``, the part of ``C_l^{psi psi}`` before ``P_R(k)``.

    Inputs:

    * ``phi``: the Weyl potential channel ``(n_c, n_k, n_tau)``.
    * ``shift_idx``, ``shift_w``: each cosmology's stencil onto the shared
      ``chi`` grid (:func:`discocmb.transfer.lagrange_shift_stencil`).
    * ``chi``: the shared grid ``(n_chi,)``.
    * ``chi_star``: distance to each visibility peak ``(n_c,)``.
    * ``operator``: :func:`lensing_potential_operator` for the exact multipoles.
    * ``limber_weights``, ``k_chi``: :func:`limber_interpolation` and
      :func:`limber_wavenumbers` for the Limber multipoles.

    ``theta_psi`` is the exact transfer function ``Theta_l(k)`` at the solved
    modes, ``(n_c, n_ell_exact, n_k)``. ``limber`` is the Limber integrand
    ``S(k_chi, chi)^2 dchi / (k_chi^3 chi^2)``, ``(n_c, n_ell_limber,
    n_chi)``, zero where ``chi <= 0``.
    :func:`lensing_potential_cl_from_transfer` multiplies both by ``P_R``.
    """

    hi = jax.lax.Precision.HIGHEST
    shifted = shift_onto_chi(phi, shift_idx, shift_w)  # (c, k, chi)
    kernel = _lensing_kernel(chi, chi_star)
    source = -2.0 * kernel[:, None, :] * shifted  # (c, k, chi)
    theta = jnp.einsum(
        "ckj,lkj->clk", source.astype(operator.dtype), operator, precision=hi
    ).astype(jnp.float64)
    source_at = jnp.einsum(
        "ljk,ckj->clj", limber_weights, source.astype(jnp.float32), precision=hi
    ).astype(jnp.float64)
    chi_safe = jnp.where(chi > 0.0, chi, 1.0)
    dchi = trapezoid_weights(chi, jnp)
    limber = jnp.where(
        chi > 0.0,
        source_at**2 / (k_chi[None] ** 3 * chi_safe[None, None] ** 2) * dchi,
        0.0,
    )
    return theta, limber


@jax.jit
def lensing_potential_cl_from_transfer(
    theta_psi, limber, k_chi, k_values, n_s, A_s, k_pivot
):
    """Return ``C_l^{psi psi}``, ``(n_c, n_ell_exact + n_ell_limber)``, from :func:`lensing_potential_transfer` and ``P_R(k)``."""

    primordial = primordial_spectrum(k_values, n_s, A_s, k_pivot)
    # Exact projection: C_l = 4 pi int dln k P_R(k) Theta_l(k)^2.
    w = trapezoid_weights(jnp.log(k_values), jnp)
    cl_exact = 4.0 * jnp.pi * jnp.einsum("clk,ck,k->cl", theta_psi**2, primordial, w)
    # Limber: C_l = 2 pi^2 int dchi S(k_chi, chi)^2 P_R(k_chi) / (k_chi^3 chi^2), k_chi = (l + 1/2) / chi.
    p_r = primordial_spectrum(k_chi, n_s, A_s, k_pivot)
    cl_limber = 2.0 * jnp.pi**2 * jnp.sum(limber * p_r, axis=-1)
    return jnp.concatenate([cl_exact, cl_limber], axis=1)


@array_cache
def lensing_operators(ells, k_sorted, tau_save, tau0_ref, dtype):
    """Return the exact table, the Limber weights and the Limber ``k_chi``, cached per grid set.

    Multipoles below :data:`LIMBER_LMIN` use the exact projection, the rest
    Limber. ``ells`` must be sorted, and every array must be a NumPy array
    because the cache keys on them.
    """

    exact, limber = ells[ells < LIMBER_LMIN], ells[ells >= LIMBER_LMIN]
    chi = tau0_ref - tau_save
    return (
        lensing_potential_operator(exact, k_sorted, tau_save, tau0_ref, dtype),
        limber_interpolation(limber, k_sorted, chi),
        jnp.asarray(limber_wavenumbers(limber, chi)),
    )


# ---------------------------------------------------------------------------
# 2. Lensing of the temperature spectrum
# ---------------------------------------------------------------------------


class LensingTables(NamedTuple):
    """Gauss-Legendre nodes and the Wigner d-functions on them, for :func:`lensed_spectra`.

    Each table holds ``d^l_{mm'}(beta)`` for ``l = 0..lmax`` on the nodes
    ``x = cos(beta)``, shape ``(lmax + 1, n)``. The name spells ``m m'``, with
    ``m`` standing for a minus sign.
    """

    lmax: int
    x: jax.Array  # (n,) cos(beta)
    weights: jax.Array  # (n,)
    d00: jax.Array
    d11: jax.Array
    d1m1: jax.Array
    d2m2: jax.Array
    d22: jax.Array
    d13: jax.Array
    d04: jax.Array
    d3m3: jax.Array
    d4m4: jax.Array
    d20: jax.Array
    d1m3: jax.Array
    d2m4: jax.Array


D_TABLES = (
    (0, 0),
    (1, 1),
    (1, -1),
    (2, -2),
    (2, 2),
    (1, 3),
    (0, 4),
    (3, -3),
    (4, -4),
    (2, 0),
    (1, -3),
    (2, -4),
)
"""``(m, m')`` of each d-function table in :class:`LensingTables`, in field order."""


def _wigner_d_start(m, mp, x):
    """Return ``d^j_{m m'}(beta)`` at the lowest ``j = max(|m|, |m'|)`` from the closed form.

    ``d^j_{j mu} = sqrt((2j)! / ((j+mu)! (j-mu)!)) cos^{j+mu}(beta/2) (-sin(beta/2))^{j-mu}``,
    mapped to ``m, m'`` by ``d^j_{m m'} = (-1)^{m-m'} d^j_{m' m} = d^j_{-m -m'}``.
    """

    from math import factorial

    sign = 1.0
    if abs(mp) > abs(m):
        sign = (-1.0) ** (m - mp)
        m, mp = mp, m
    if m < 0:
        m, mp = -m, -mp
    j, mu = m, mp
    coefficient = np.sqrt(factorial(2 * j) / (factorial(j + mu) * factorial(j - mu)))
    cos_half = np.sqrt(0.5 * (1.0 + x))
    sin_half = np.sqrt(0.5 * (1.0 - x))
    return sign * coefficient * cos_half ** (j + mu) * (-sin_half) ** (j - mu)


def _wigner_d(m, mp, x, lmax):
    """Return ``d^l_{m m'}(beta)`` for ``l = 0..lmax`` on ``x = cos(beta)`` by upward recursion."""

    x = np.asarray(x, dtype=np.float64)
    out = np.zeros((lmax + 1, len(x)))
    l0 = max(abs(m), abs(mp))
    if l0 > lmax:
        return out
    out[l0] = _wigner_d_start(m, mp, x)
    prev = np.zeros_like(x)
    if l0 == 0:  # the recursion divides by l, so seed it with d^1_00 = cos(beta)
        if lmax == 0:
            return out
        out[1], prev, l0 = x, out[0], 1
    for l in range(l0, lmax):
        num = (2 * l + 1) * (l * (l + 1) * x - m * mp) * out[l] - (l + 1) * np.sqrt(
            (l * l - m * m) * (l * l - mp * mp)
        ) * prev
        den = l * np.sqrt(((l + 1) ** 2 - m * m) * ((l + 1) ** 2 - mp * mp))
        prev = out[l]
        out[l + 1] = num / den
    return out


def lensing_tables(
    lmax: int, n_points: int | None = None, dtype=jnp.float32
) -> LensingTables:
    """Tabulate the quadrature and d-functions for :func:`lensed_spectra`.

    ``n_points`` sets the Gauss-Legendre nodes in ``cos(beta)``. The default,
    ``lmax + 1024``, resolves the correlation functions to better than 1e-4 of
    the spectra. At ``lmax ~ 3000`` this takes several seconds on the host and
    ~600 MB on the device in float32.
    """

    n = n_points or (lmax + 1024)
    x, w = np.polynomial.legendre.leggauss(n)
    return LensingTables(
        lmax,
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(w, dtype=jnp.float64),
        *(jnp.asarray(_wigner_d(m, mp, x, lmax), dtype=dtype) for m, mp in D_TABLES),
    )


def cl_on_every_ell(ells, cl, lmax: int, power: int = 1):
    """Return ``C_l`` at every ``l = 0..lmax``, ``(n_c, lmax + 1)``, from samples ``(n_c, n_ell)`` at sorted ``ells``.

    :func:`lensed_spectra` needs every multipole, but the pipeline computes a
    few dozen. This fits a natural cubic spline in ``ln l`` to
    ``[l(l+1)]^power C_l``, which is nearly flat for ``power = 1`` (CMB
    spectra) and ``power = 2`` (lensing potential). The result is zero below
    ``ells[0]``, and ``ells[-1]`` must reach ``lmax``. Gaps where the spectrum
    curves sharply cost accuracy. Sampling TT at ``l = 2, 4`` misses ``l = 3``
    by 5e-3, so sample every multipole there.
    """

    from discoeb.spline_interpolation import spline_interpolation

    ells = np.asarray(ells, dtype=np.float64)
    if ells[-1] < lmax:
        raise ValueError(f"samples end at l = {ells[-1]:.0f}, below lmax = {lmax}")
    ell = np.arange(lmax + 1, dtype=np.float64)
    inside = ell >= ells[0]
    weight = (ells * (ells + 1.0)) ** power
    spline = spline_interpolation(
        jnp.log(jnp.asarray(ells)), (jnp.asarray(cl, dtype=jnp.float64) * weight).T
    )
    values = spline.evaluate(
        jnp.log(jnp.asarray(np.where(inside, ell, ells[0])))
    )  # (lmax + 1, n_c)
    unweight = np.where(inside, ell * (ell + 1.0), 1.0) ** power
    return jnp.where(inside[None, :], (values / unweight[:, None]).T, 0.0)


class LensedSpectra(NamedTuple):
    """Lensed spectra from :func:`lensed_spectra`, each ``(n_c, lmax + 1)``."""

    tt: jax.Array
    te: jax.Array
    ee: jax.Array
    bb: jax.Array


def _lensed_one(cl_tt, cl_te, cl_ee, cl_pp, tables: LensingTables):
    """Return lensed ``(TT, TE, EE, BB)`` for ``l = 0..lmax`` of one cosmology in float64."""

    hi = jax.lax.Precision.HIGHEST
    dtype = tables.d00.dtype
    ell = jnp.arange(
        tables.d00.shape[0], dtype=jnp.float64
    )  # 0..lmax (a static shape, unlike tables.lmax under jit)
    llp1 = ell * (ell + 1.0)
    norm = (2.0 * ell + 1.0) / (4.0 * jnp.pi)
    # Deflection correlation functions on the quadrature nodes.
    a = (norm * llp1 * cl_pp).astype(dtype)
    c_gl = jnp.einsum("l,ln->n", a, tables.d11, precision=hi)
    c_gl2 = jnp.einsum("l,ln->n", a, tables.d1m1, precision=hi)
    sigma2 = (jnp.sum(a.astype(jnp.float64)) - c_gl.astype(jnp.float64)).astype(dtype)
    damping = jnp.exp(
        -0.5 * llp1.astype(dtype)[:, None] * sigma2[None, :]
    )  # F_l, (l, n)
    # The anisotropic expansion parameter is c_l = h_l C_gl,2 with h_l = l(l+1)/2.
    # Each term is F_l times a power of C_gl,2(beta) times a sum over l.
    h = 0.5 * llp1
    b_t, b_x, b_e = (norm * cl_tt, norm * cl_te, norm * cl_ee)

    def term(b, coefficient, d):
        return jnp.einsum(
            "l,ln,ln->n", (b * coefficient).astype(dtype), damping, d, precision=hi
        )

    xi = (
        term(b_t, 1.0, tables.d00)
        + c_gl2 * term(b_t, h, tables.d1m1)
        + c_gl2**2
        * (term(b_t, h**2 / 4.0, tables.d00) + term(b_t, h**2 / 4.0, tables.d2m2))
    )
    xi_plus = (
        term(b_e, 1.0, tables.d22)
        + c_gl2 * term(b_e, h, tables.d13)
        + c_gl2**2
        * (term(b_e, h**2 / 4.0, tables.d22) + term(b_e, h**2 / 4.0, tables.d04))
    )
    xi_minus = (
        term(b_e, 1.0, tables.d2m2)
        + c_gl2 * (term(b_e, h / 2.0, tables.d1m1) + term(b_e, h / 2.0, tables.d3m3))
        + c_gl2**2
        * (
            term(b_e, h**2 / 4.0, tables.d2m2)
            + term(b_e, h**2 / 8.0, tables.d00)
            + term(b_e, h**2 / 8.0, tables.d4m4)
        )
    )
    xi_cross = (
        term(b_x, 1.0, tables.d20)
        + c_gl2 * (term(b_x, h / 2.0, tables.d11) + term(b_x, h / 2.0, tables.d1m3))
        + c_gl2**2
        * (term(b_x, 3.0 * h**2 / 8.0, tables.d20) + term(b_x, h**2 / 8.0, tables.d2m4))
    )

    def back(xi_n, d):
        integrand = (tables.weights * xi_n.astype(jnp.float64)).astype(dtype)
        return (
            2.0
            * jnp.pi
            * jnp.einsum("ln,n->l", d, integrand, precision=hi).astype(jnp.float64)
        )

    tt = back(xi, tables.d00)
    te = back(xi_cross, tables.d20)
    plus = back(xi_plus, tables.d22)
    minus = back(xi_minus, tables.d2m2)
    return tt, te, 0.5 * (plus + minus), 0.5 * (plus - minus)


@partial(jax.jit, static_argnames=("chunk",))
def lensed_spectra(
    cl_tt, cl_te, cl_ee, cl_pp, tables: LensingTables, chunk: int = 4
) -> LensedSpectra:
    """Return lensed ``TT, TE, EE, BB`` for a batch as a :class:`LensedSpectra` of ``(n_c, lmax + 1)``.

    ``cl_tt``, ``cl_te``, ``cl_ee`` (unlensed) and ``cl_pp`` (lensing
    potential) are ``(n_c, lmax + 1)``, indexed by ``l`` from 0. The function
    ignores the monopole and dipole. :func:`cl_on_every_ell` fills these
    arrays from the sampled output of
    :func:`discocmb.main.compute_cl_power_spectrum_batch`. The unlensed
    spectra must run a few hundred multipoles past the highest lensed one you
    need. It handles ``chunk`` cosmologies at a time, since each needs an
    ``(lmax + 1, n_points)`` array of damping factors.
    """

    spectra = [
        jnp.asarray(c, dtype=jnp.float64).at[:, :2].set(0.0)
        for c in (cl_tt, cl_te, cl_ee, cl_pp)
    ]
    n_c, n_l = spectra[0].shape
    pad = (-n_c) % chunk
    spectra = [jnp.pad(c, ((0, pad), (0, 0))).reshape(-1, chunk, n_l) for c in spectra]
    one = jax.vmap(_lensed_one, in_axes=(0, 0, 0, 0, None))
    out = jax.lax.map(lambda args: one(*args, tables), tuple(spectra))
    return LensedSpectra(*(c.reshape(-1, n_l)[:n_c] for c in out))
