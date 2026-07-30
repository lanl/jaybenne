#!/usr/bin/env python
# ========================================================================================
#  (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

import sys

sys.dont_write_bytecode = True

import os
import regression_test as rt
import numpy as np
from scipy.special import erf

# -- constants
hp = 6.626e-27 #[erg-s]
kb = 1.381e-16 #[erg/K]
cl = 2.998e10  #[cm/s]

# -- helper functions
def planck(T, nu):
    x = hp * nu / (kb * T)
    efac = np.exp(-x)
    return x * x * x * efac / (1.0 - efac)

class Plaw_Opac:
    def __init__(self, kappa0_, rho_exp_, temp_exp_, nu_exp_,
                 nu_ref_, nu_off_, rho_ref_, rho_off_,
                 temp_ref_, temp_off_):
        self.kappa0 = kappa0_
        self.rho_exp = rho_exp_
        self.temp_exp = temp_exp_
        self.nu_exp = nu_exp_
        self.nu_ref = nu_ref_
        self.nu_off = nu_off_
        self.rho_ref = rho_ref_
        self.rho_off = rho_off_
        self.temp_ref = temp_ref_
        self.temp_off = temp_off_

    def plaw_opac(self, rho, T, nu):
        xrho = (rho + self.rho_off) / self.rho_ref
        xT = (T + self.temp_off) / self.temp_ref
        xnu = (nu + self.nu_off) / self.nu_ref
        return self.kappa0 * xrho**self.rho_exp * xT**self.temp_exp * xnu**self.nu_exp

# -- parser
parser = rt.get_default_parser()
args = parser.parse_args()

modified_inputs = {}
modified_inputs["parthenon/mesh/nx1"] = 128
modified_inputs["parthenon/meshblock/nx1"] = 128

# -- frequency grid
n_nubins = 16
numin = 1.e12 #[Hz]
numax = 1.e17 #[Hz]
dlnu = np.log(numax / numin) / n_nubins
nu_grid = np.array([numin * np.exp((ig + 0.5) * dlnu) for ig in range(n_nubins)])
dnu = nu_grid * dlnu

# -- Analytic solution
nur = nu_grid[12]
nuo = nu_grid[8]
pops = Plaw_Opac(0.75e3, 0.0, 0.0, -0.25, nur, nuo, 1.0, 0.0, 1.0, 0.0)
tau = 3.0 * pops.plaw_opac(1.0, 1.0, nu_grid) / cl # = 3 * sigma_t / c
T0 = 1e5
plnk = planck(T0, nu_grid) * dnu
plnk /= np.sum(plnk)
ur0 = 7.5646e5 * plnk
shift = 0.5


def urnu_solution(t, x, y, z, ig):
    return (
        ur0[ig]
        / 2.0
        * (
            erf(((x + shift) + 0.5) / (2.0 * np.sqrt(t / tau[ig])))
            - erf(((x + shift) - 0.5) / (2.0 * np.sqrt(t / tau[ig])))
        )
    )

def ur_solution(t, x, y, z):
    ur = sum([urnu_solution(t, x, y, z, ig) for ig in range(n_nubins)])
    return ur

code = rt.analytic_comparison(
    args=args,
    variables=["field.jaybenne.energy_tally"],
    solutions=[ur_solution],
    modified_inputs=modified_inputs,
    tolerance=0.05,
)

sys.exit(code)
