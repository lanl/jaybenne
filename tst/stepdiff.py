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


parser = rt.get_default_parser()
args = parser.parse_args()

modified_inputs = {}
modified_inputs["parthenon/mesh/nx1"] = 128
modified_inputs["parthenon/meshblock/nx1"] = 128

# -- Analytic solution
tau = 1.000692e-7
ur0 = 7.5646e5
shift = 0.5


def ur_solution(t, x, y, z):
    return (
        ur0
        / 2.0
        * (
            erf(((x + shift) + 0.5) / (2.0 * np.sqrt(t / tau)))
            - erf(((x + shift) - 0.5) / (2.0 * np.sqrt(t / tau)))
        )
    )


code = rt.analytic_comparison(
    args=args,
    variables=["field.jaybenne.energy_tally"],
    solutions=[ur_solution],
    modified_inputs=modified_inputs,
    tolerance=0.05,
)

sys.exit(code)
