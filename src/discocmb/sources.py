"""CMB temperature and polarization source functions.

Line-of-sight integration (Seljak & Zaldarriaga 1996, :mod:`discocmb.transfer`)
builds today's anisotropy from a few local sources. They combine the
perturbation state with the visibility ``g = kappa' exp(-kappa)``:

    * ``S``: the monopole (Sachs-Wolfe) and polarization correction times
      ``g``, plus the integrated Sachs-Wolfe term ``2 Phi' exp(-kappa)``.
    * ``S_j1``: the Doppler term ``g (sigma + v_b)``.
    * ``S_j2``: the quadrupole and polarization term.

They multiply ``j_l(x)``, ``j_l'(x)`` and ``j_l''(x)`` with ``x = k (tau_0 -
tau)``. The E-mode transfer function reuses ``S_j2`` against ``j_l(x) / x^2``
(CAMB's ``sources(2)``).

**Where the sources live.** :class:`LineOfSightSources` adds them to the
DISCO-EB model, and :func:`solve_sources` asks :func:`~discoeb.solve_eb` to
keep them, plus the opacity and optical depth, at every save time
(:data:`SOURCE_KEEP`). The kernel returns six numbers per save and trajectory
instead of the full state, so a whole batch fits in one launch.
:func:`assemble_sources` takes the ISW term's ``phi'`` as a centered
difference of the saved potential. The analytic derivative amplifies the
solver's state error tenfold at ``l = 2``. :func:`visibility_on_grid` splines
the opacity and integrates the spline for ``kappa``. A trapezoid on the save
grid overshot ``kappa'`` by 0.2 percent through recombination and lowered
``C_l`` by 0.5 percent.

**Polarization.** At low ``k`` the polarization source sits below the solve's
absolute tolerance. :func:`polarization_error_weights` scales up the photon
quadrupole and ``E_2`` in the error norm by ``1/k``. Without it the low
multipoles of EE and TE come out 10 to 40 percent low.
"""

from __future__ import annotations

import discoeb as eb
import jax
import jax.numpy as jnp
import numpy as np
from discoeb import Component

SOURCE_CHANNELS = 6
"""Numbers the solve keeps per save (:data:`SOURCE_KEEP`).

``m0, m1, m2, phi, kappa', kappa``: the three unweighted sources, the Weyl
potential, the opacity and the optical depth from ``tau_start``.
:func:`weight_channels` turns them into ``S_noisw, S_j1, S_j2, phi,
exp(-kappa)``."""

WEIGHTED_CHANNELS = 5
"""Numbers per save after weighting: ``S_noisw, S_j1, S_j2, phi, exp(-kappa)``."""


class LineOfSightSources(Component):
    """CAMB-form temperature and polarization sources as a model component.

    It provides the Weyl potential ``phi = -(dgrho + 3 aH dgq / k + dgpi) /
    (2 k^2)`` and the unweighted parts of the three sources, without the ISW
    term: ``monopole + 5/8 Pi``, ``sigma + v_b`` and ``15/8 Pi``, with ``Pi``
    the scattering source ``polter``. Every species the layout enables
    contributes through the sums ``dgrho``, ``dgq`` and ``dgpi``. The batch
    solve and :func:`~discocmb.main.cl_power_spectrum_jax` share these
    equations.
    """

    def phi(self, dgrho, dgrho_streaming, dgq, dgpi, adotoa, k, inv_k2):
        """Return the Weyl potential, including radiation under the streaming approximation."""
        return -(dgrho + dgrho_streaming + 3.0 * dgq * adotoa / k + dgpi) * 0.5 * inv_k2

    def monopole(self, etak, k, phi, clxg):
        return -etak / k + 2.0 * phi + 0.25 * clxg

    def m0(self, monopole, polter):
        return monopole + 0.625 * polter

    def m1(self, sigma, vb):
        return sigma + vb

    def m2(self, polter):
        return 1.875 * polter


def source_model(layout):
    """Return the joint model of ``layout`` with :class:`LineOfSightSources` added."""

    return eb.joint_model(layout, extra=(LineOfSightSources,))


POLARIZATION_WEIGHT_K = 1.0
"""Wave number (Mpc^-1) below which the error norm scales up the photon
quadrupole and ``E_2`` by ``POLARIZATION_WEIGHT_K / k``."""

POLARIZATION_WEIGHT_MAX = 1.0e4
"""Upper limit of that weight."""


def polarization_error_weights(cosmology) -> dict:
    """Return ``{state name: weight of k}``, error weights that resolve the small low-``k`` sources.

    The E-mode source is ``g polter`` with ``polter = pig/10 + 3/5 E_2``. Well
    below the acoustic scale both moments stay near the solve's ``atol =
    1e-4`` through recombination, so step-size control ignores them and the
    low multipoles of EE and TE come out 10 to 40 percent low. TT does not
    suffer because its sources are of order one.

    The weight ``clip(POLARIZATION_WEIGHT_K / k, 1, POLARIZATION_WEIGHT_MAX)``
    divides the tolerance of the photon quadrupole ``theta_2`` and ``E_2``
    (``e_0``). It slows only the low-``k`` trajectories, which take the fewest
    steps. A batch of 128 costs 10 percent more (0.64 s against 0.58 s on an
    RTX 4070 SUPER), and EE matches CLASS to 0.35 percent from ``l = 2``. A
    scalar tolerance of 1e-5 reaches the same accuracy at 1.8 times the cost.

    The massive-neutrino monopole, dipole and quadrupole of each momentum bin
    (``psi_q_0`` to ``psi_q_2``) get the same weight. They are just as small
    at low ``k`` and feed the shear that drives the photon quadrupole. Without
    the weight a 0.3 eV neutrino leaves EE at ``l = 2`` 2 percent low.

    The keys are the cosmology's state names (:func:`discoeb.state_names`),
    the form :func:`discoeb.solve_eb` takes as ``error_weights``.
    """

    def weight(k):
        return np.clip(
            POLARIZATION_WEIGHT_K / np.asarray(k), 1.0, POLARIZATION_WEIGHT_MAX
        )

    names = eb.state_names(cosmology)
    weighted = ["theta_2", "e_0"] + [
        n for n in names if n.startswith("psi_") and int(n.rsplit("_", 1)[1]) < 3
    ]
    return {n: weight for n in weighted}


# =============================================================================
# The visibility function
# =============================================================================


def _grad_last(y, x):
    """Return ``dy/dx`` along the last axis by centered differences."""

    mid = (y[..., 2:] - y[..., :-2]) / (x[..., 2:] - x[..., :-2])
    first = (y[..., 1:2] - y[..., :1]) / (x[..., 1:2] - x[..., :1])
    last = (y[..., -1:] - y[..., -2:-1]) / (x[..., -1:] - x[..., -2:-1])
    return jnp.concatenate([first, mid, last], axis=-1)


VISIBILITY_REFINEMENT = 16
"""Sub-intervals per thermodynamics-table interval in the optical-depth integral."""


def visibility_on_grid(tau_grid, opacity, tau_save, tau0):
    """Return ``(kappa', g, exp(-kappa))`` on the save grid from a tabulated opacity.

    A cubic spline of ``ln kappa'`` in ``ln tau`` interpolates the opacity,
    and integrating it on a :data:`VISIBILITY_REFINEMENT`-fold finer grid
    gives ``kappa(tau) = int_tau^{tau0} kappa' dtau'``. A trapezoid on the save
    grid is not accurate enough. ``kappa'`` drops by ``e`` every ~15 Mpc
    through recombination, so on a 3 Mpc grid the trapezoid overshoots
    ``kappa`` by ~3e-3 near ``kappa ~ 1`` and lowers every ``C_l`` by 0.5
    percent. Past ``tau0`` both ``g`` and ``exp(-kappa)`` are zero, which
    zeroes the sources there.
    """

    from scipy.interpolate import CubicSpline

    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64)
    log_spline = CubicSpline(np.log(tau_grid), np.log(opacity))

    n = len(tau_grid)
    fine = np.exp(
        np.linspace(
            np.log(tau_grid[0]), np.log(tau0), VISIBILITY_REFINEMENT * (n - 1) + 1
        )
    )
    kappa_prime_fine = np.exp(log_spline(np.log(fine)))
    kappa_fine = np.concatenate(
        [
            np.cumsum(
                (0.5 * (kappa_prime_fine[1:] + kappa_prime_fine[:-1]) * np.diff(fine))[
                    ::-1
                ]
            )[::-1],
            [0.0],
        ]
    )

    live = tau_save <= tau0
    tau_live = np.minimum(tau_save, tau0)
    kappa_prime = np.where(live, np.exp(log_spline(np.log(tau_live))), 0.0)
    kappa = np.interp(tau_live, fine, kappa_fine)
    emk = np.where(live, np.exp(-kappa), 0.0)
    return kappa_prime, kappa_prime * emk, emk


# =============================================================================
# Visibility weighting and the ISW term
# =============================================================================


def weight_channels(channels, tau_save, tau0):
    """Turn the six kept channels into ``S_noisw, S_j1, S_j2, phi, exp(-kappa)``.

    ``channels`` is ``(n_cosmo, 6, n_k, n_tau)`` and ``tau0`` holds the
    conformal ages. The optical depth to today is ``kappa(tau_0) -
    kappa(tau)``, with ``kappa(tau_0)`` read at the last save. The large
    early-time part cancels in the difference, so ``exp(-kappa)`` keeps the
    accuracy of the kernel's increments. The sources vanish past each
    cosmology's ``tau_0``.
    """

    m0, m1, m2, phi, kappa_prime, kappa = (
        channels[:, i] for i in range(SOURCE_CHANNELS)
    )
    kappa_end = kappa[:, :, -1:]
    live = jnp.asarray(tau_save)[None, None, :] <= jnp.asarray(tau0)[:, None, None]
    emk = jnp.where(live, jnp.exp(-(kappa_end - kappa)), 0.0)
    g = kappa_prime * emk
    return jnp.stack([g * m0, g * m1, g * m2, phi, emk], axis=1)


@jax.jit
def channels_from_solution(recorded, tau_save, tau0):
    """Return ``(chi_star, weighted channels)`` from the kept ``(n_cosmo, 6, n_k, n_tau)``.

    Applies :func:`weight_channels` and finds ``chi_star = tau0 - tau_star``,
    the distance to each cosmology's visibility peak. The first wave mode
    suffices for the peak since every mode shares the recombination history.
    Jitting it as one program avoids a dozen eager passes over a gigabyte of
    sources.
    """

    weighted = weight_channels(recorded, tau_save, tau0)
    g = recorded[:, 4, 0] * weighted[:, 4, 0]  # kappa' exp(-kappa), zero past tau0
    return tau0 - tau_save[jnp.argmax(g, axis=1)], weighted


def assemble_sources(channels, tau):
    """Return ``(S, S_j1, S_j2)`` as ``(..., 3, n_k, n_tau)`` from the weighted channels.

    ``channels`` is ``(..., 5, n_k, n_tau)``. Centered differences of ``phi``
    give the ISW term.
    """

    s_noisw, s_j1, s_j2, phi, emk = (
        channels[..., i, :, :] for i in range(WEIGHTED_CHANNELS)
    )
    s_j0 = s_noisw + 2.0 * _grad_last(phi, jnp.asarray(tau)) * emk
    return jnp.stack([s_j0, s_j1, s_j2], axis=-3)


# =============================================================================
# The sources, kept by the Einstein-Boltzmann solve
# =============================================================================
#
# discoeb.solve_eb evaluates any quantity of its model at every save time from
# the dense-output state. The kernel keeps the sources and returns six numbers
# per save instead of the full state, and the projection continues on the
# device.


SOURCE_KEEP = ("m0", "m1", "m2", "phi", "opacity", "kappa")
"""What :func:`solve_sources` keeps at every save, in channel order.

The three unweighted CAMB-form sources without ISW, the Weyl potential, the
opacity ``kappa'`` and the optical depth ``kappa``. It leaves out ``phi'``
on purpose. Late-time ``phi'`` is a small residual of large terms that cancel,
so the analytic derivative amplifies the solver's state error. Differencing
saved potentials cancels that error, which :func:`assemble_sources` does."""


def solve_sources(cosmologies, k_values, tau_save, *, inputs=None, **solve_kwargs):
    """Solve a batch and return its source channels, ``values`` of ``(n_cosmo, 6, n_k, n_tau)``.

    Runs one :func:`discoeb.solve_eb` launch that keeps :data:`SOURCE_KEEP`.
    ``tau_save`` starts at ``TAU_START``. The launch uses
    :func:`polarization_error_weights` unless ``solve_kwargs`` has
    ``error_weights``. Pass DISCO-EB's directed ``inputs`` to differentiate
    with respect to them (:func:`~discocmb.main.cl_power_spectrum_jax`).
    """

    cosmologies = tuple(cosmologies)
    if solve_kwargs.get("error_weights") is None:
        solve_kwargs["error_weights"] = polarization_error_weights(cosmologies[0])
    return eb.solve_eb(
        cosmologies,
        k_values,
        tau_save,
        keep=SOURCE_KEEP,
        extra=(LineOfSightSources,),
        inputs=inputs,
        **solve_kwargs,
    )
