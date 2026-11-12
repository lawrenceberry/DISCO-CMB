"""DISCO-CMB: CMB angular power spectra from DISCO-EB's Einstein-Boltzmann solve.

The computation has four steps, one module each, and :mod:`discocmb.main`
chains them:

1. :mod:`discocmb.sources`: the line-of-sight sources, which DISCO-EB's
   kernel evaluates at every save time.
2. :mod:`discocmb.transfer`: line-of-sight integration into the temperature
   and E-mode transfer functions.
3. :mod:`discocmb.spectra`: the unlensed TT, TE and EE spectra.
4. :mod:`discocmb.lensing`: the lensing potential and the lensed spectra.

:mod:`discocmb.grids` builds the grids the steps share. The package talks to
DISCO-EB only through :mod:`discoeb`.
"""

from .grids import CMBGrids, cmb_grids, cmb_k_grid
from .lensing import cl_on_every_ell, lensed_spectra, lensing_tables
from .main import (
    cl_power_spectrum_jax,
    compute_cl_power_spectrum_batch,
    compute_transfer_batch,
)
from .spectra import Spectra

__all__ = [
    "CMBGrids",
    "Spectra",
    "cl_on_every_ell",
    "cl_power_spectrum_jax",
    "cmb_grids",
    "cmb_k_grid",
    "compute_cl_power_spectrum_batch",
    "compute_transfer_batch",
    "lensed_spectra",
    "lensing_tables",
]
