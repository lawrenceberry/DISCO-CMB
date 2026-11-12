"""CMB power spectra end to end, for batches of cosmologies.

1. **Sources** (:mod:`discocmb.sources`): one DISCO-EB launch
   (:func:`~discocmb.sources.solve_sources`) integrates every wave mode of
   every cosmology and keeps the source channels at each save time. The
   device then applies the visibility weighting.
2. **Line-of-sight integration** (:mod:`discocmb.transfer`): shift the
   sources onto a shared ``chi`` grid, refine them in ``ln k`` and project
   them against Bessel tables into ``Theta_l(k)`` and ``E_l(k)``.
3. **Power spectra** (:mod:`discocmb.spectra`): TT, TE and EE ``C_l`` and
   ``D_l`` from the transfer functions and the primordial spectrum.
4. **Lensing** (:mod:`discocmb.lensing`): on request, ``C_l^{psi psi}`` from
   the saved Weyl potential. :func:`discocmb.lensing.lensed_spectra` then
   lenses the spectra.

:mod:`discocmb.grids` supplies the grids.
"""

from __future__ import annotations

import discoeb as eb
import jax.numpy as jnp
import numpy as np

from .grids import CMBGrids, cmb_grids
from .lensing import lensing_operators, lensing_potential_transfer
from .sources import channels_from_solution, solve_sources
from .spectra import Spectra, spectra_from_transfer
from .transfer import (
    TransferBatch,
    TransferGrids,
    k_band,
    lagrange_shift_stencil,
    projection_k_chunk,
    projection_operator,
    theta_ell_batch,
)


def compute_transfer_batch(
    cosmologies,
    *,
    grids: CMBGrids | None = None,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    projection_dtype=jnp.float32,
    lensing_potential_ells: np.ndarray | None = None,
    **solve_kwargs,
):
    """Return ``(TransferBatch, TransferGrids)``: everything before ``P_R(k)``.

    :func:`compute_cl_power_spectrum_batch` calls this and then
    :func:`~discocmb.spectra.spectra_from_transfer`. See it for the arguments.
    """

    cosmologies = tuple(eb.as_cosmology(c) for c in cosmologies)
    tau0_all = eb.conformal_ages(cosmologies)
    if grids is None:
        grids = cmb_grids(
            cosmologies[0],
            k_values=k_values,
            ells=ells,
            n_save=n_save,
            n_k_fine=n_k_fine,
            k_fine_values=k_fine_values,
            tau_end=float(np.max(tau0_all)),
        )
    k_values, ells, k_fine, tau_save = grids
    tau0_ref = grids.tau_end
    if np.max(tau0_all) > tau0_ref * (1.0 + 1.0e-12):
        raise ValueError(
            f"a cosmology's tau0 = {np.max(tau0_all):.1f} Mpc lies past the save grid's "
            f"end {tau0_ref:.1f} Mpc: build the grids with a larger tau_end_margin"
        )

    # 1. Sources: one DISCO-EB launch keeps the source channels at every save.
    # The kept names join the kernel-cache key, so like the right-hand side
    # they compile once per layout and no batch uploads anything extra.
    k_sorted = np.sort(np.asarray(k_values, dtype=np.float64))
    solution = solve_sources(cosmologies, k_sorted, tau_save, **solve_kwargs)
    # (n_cosmo, 6, n_k, n_tau), then the visibility weighting
    chi_star, channels = channels_from_solution(
        solution.values, jnp.asarray(tau_save), jnp.asarray(tau0_all)
    )
    del solution

    # 2. Line-of-sight integration onto the shared chi grid.
    interp, bessel, bessel_e, lnk_weights = projection_operator(
        ells, k_sorted, k_fine, tau_save, tau0_ref, projection_dtype
    )
    ells_d = jnp.asarray(ells, dtype=jnp.float64)
    k_fine_d = jnp.asarray(k_fine)
    tau_d = jnp.asarray(tau_save)
    # Build the per-cosmology arrays (chi shifts, k bands, primordial
    # amplitudes) as one host array and one device transfer for the batch.
    # An eager per-cosmology ``jnp`` loop here cost 80 ms of host time a batch
    # and stalled on the running kernel.
    chi_max = float(tau0_ref - tau_save[0])
    shift_idx, shift_w = lagrange_shift_stencil(tau_save, tau0_all - tau0_ref)
    shift_idx, shift_w = jnp.asarray(shift_idx), jnp.asarray(shift_w)
    bands = k_band(ells_d, k_fine_d, chi_max, chi_star)
    T_cmb = jnp.asarray([float(c.T_cmb) for c in cosmologies])
    k_pivot = float(cosmologies[0].k_pivot)

    # One call for the whole batch, chunked over fine wave modes. That reads
    # the Bessel table once and keeps each chunk within the memory budget.
    # Chunking over cosmologies read the table once per group (60 ms for a
    # batch of 128 against 37).
    theta, theta_e = theta_ell_batch(
        channels,
        tau_d,
        shift_idx,
        shift_w,
        bands,
        interp,
        bessel,
        bessel_e,
        k_chunk=projection_k_chunk(
            len(cosmologies), len(k_fine), len(tau_save), projection_dtype
        ),
    )
    if lensing_potential_ells is None:
        transfer = TransferBatch(theta, theta_e, None, None, T_cmb)
        tgrids = TransferGrids(ells, k_fine_d, lnk_weights, k_pivot, None, None, None)
        return transfer, tgrids

    # 4. Lensing potential transfer from the saved Weyl potential.
    ells_pp = np.asarray(lensing_potential_ells, dtype=np.int64)
    operator, limber_weights, k_chi = lensing_operators(
        ells_pp, k_sorted, tau_save, tau0_ref, projection_dtype
    )
    theta_psi, limber = lensing_potential_transfer(
        channels[:, 3],
        shift_idx,
        shift_w,
        jnp.asarray(tau0_ref - tau_save),
        chi_star,
        operator,
        limber_weights,
        k_chi,
    )
    transfer = TransferBatch(theta, theta_e, theta_psi, limber, T_cmb)
    tgrids = TransferGrids(
        ells, k_fine_d, lnk_weights, k_pivot, ells_pp, jnp.asarray(k_sorted), k_chi
    )
    return transfer, tgrids


def compute_cl_power_spectrum_batch(
    cosmologies,
    *,
    grids: CMBGrids | None = None,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    projection_dtype=jnp.float32,
    lensing_potential_ells: np.ndarray | None = None,
    **solve_kwargs,
):
    """Return ``(ells, C_l, D_l)`` for a batch, each a :class:`~discocmb.spectra.Spectra` of ``(N, n_ell)``.

    ``C_l`` holds the unlensed ``tt``, ``te`` and ``ee`` spectra in units of
    ``T_cmb^2``, and ``D_l = l(l+1) C_l / 2pi`` the same in ``muK^2``. Pass
    ``lensing_potential_ells`` to get ``C_l^{psi psi}`` at those multipoles as
    a fourth element, ``(N, n_ell_psi)``.
    :func:`discocmb.lensing.lensed_spectra` turns the two into lensed TT, TE,
    EE and BB.

    One modax launch solves every wave mode of every cosmology and returns six
    numbers per save and trajectory, with no history, so only the kernel's own
    memory limits the batch size.

    ``grids`` (from :func:`~discocmb.grids.cmb_grids`) fixes the wave modes,
    multipoles, fine ``k`` grid and save grid. Build it once from a fiducial
    cosmology and every batch reuses the projection operator. Without it the
    batch's first cosmology shapes the grids from the other keyword
    arguments, the save grid ends at the batch's largest ``tau0``, and the
    operator gets rebuilt whenever the grids change.

    ``projection_dtype`` sets the precision of the projection's matrix
    products: ``float32`` (default, 2.6e-4 in ``D_l``) or ``float64`` (ten
    times slower on consumer GPUs). ``solve_kwargs`` go to the solver
    (``rtol``, ``atol``, ``first_step``, ``max_steps``).
    """

    cosmologies = tuple(eb.as_cosmology(c) for c in cosmologies)
    transfer, tgrids = compute_transfer_batch(
        cosmologies,
        grids=grids,
        k_values=k_values,
        ells=ells,
        n_save=n_save,
        n_k_fine=n_k_fine,
        k_fine_values=k_fine_values,
        projection_dtype=projection_dtype,
        lensing_potential_ells=lensing_potential_ells,
        **solve_kwargs,
    )
    # 3. Power spectra (and C_l^{psi psi}) with the primordial spectrum.
    n_s = jnp.asarray([float(c.n_s) for c in cosmologies])
    A_s = jnp.asarray([float(c.A_s) for c in cosmologies])
    cl, dl, cl_pp = spectra_from_transfer(transfer, tgrids, n_s, A_s)
    if lensing_potential_ells is None:
        return tgrids.ells, cl, dl
    return tgrids.ells, cl, dl, cl_pp


def cl_power_spectrum_jax(
    cosmology,
    densities=None,
    thermal=None,
    *,
    grids: CMBGrids | None = None,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    projection_dtype=jnp.float64,
    **solve_kwargs,
):
    """Return ``(ells, C_l, D_l)`` of one cosmology, differentiable in its inputs.

    ``C_l`` and ``D_l`` are :class:`~discocmb.spectra.Spectra` of shape
    ``(n_ell,)``. ``densities`` (``grhog, grhornomass, grhoc, grhob, grhov``)
    and ``thermal`` (``YHe, Tcmb, H0, Omegab``) make up DISCO-EB's directed
    inputs (:data:`discoeb.DIRECTED_NAMES`) and default to the cosmology's
    own. Pass them as JAX values to differentiate the spectra with respect to
    them.

    The kernel forms the derivative itself. It carries forward sensitivities
    through the hierarchy and the recombination history, and the source
    algebra sees the densities again, so nothing stores a history. The grids,
    ``tau0`` and the reionization redshift stay fixed.

    This runs :func:`compute_transfer_batch` and
    :func:`~discocmb.spectra.spectra_from_transfer` on a batch of one. The
    solve uses :func:`~discocmb.sources.polarization_error_weights` unless you
    pass ``error_weights``. The projection defaults to ``float64`` because
    ``float32`` costs 2.6e-4 in ``D_l``, too much for a derivative.
    """

    cosmology = eb.as_cosmology(cosmology)
    inputs = None
    if densities is not None or thermal is not None:
        own = jnp.asarray(eb.directed_inputs(cosmology)[0])
        inputs = jnp.concatenate(
            [
                own[:5] if densities is None else jnp.asarray(densities, jnp.float64),
                own[5:] if thermal is None else jnp.asarray(thermal, jnp.float64),
            ]
        )[None]
    transfer, tgrids = compute_transfer_batch(
        (cosmology,),
        grids=grids,
        k_values=k_values,
        ells=ells,
        n_save=n_save,
        n_k_fine=n_k_fine,
        k_fine_values=k_fine_values,
        projection_dtype=projection_dtype,
        inputs=inputs,
        **solve_kwargs,
    )
    cl, dl, _ = spectra_from_transfer(
        transfer,
        tgrids,
        jnp.asarray([float(cosmology.n_s)]),
        jnp.asarray([float(cosmology.A_s)]),
    )
    return tgrids.ells, Spectra(*(c[0] for c in cl)), Spectra(*(d[0] for d in dl))
