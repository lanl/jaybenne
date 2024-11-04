//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#ifndef JAYBENNE_PLANCK_HPP_
#define JAYBENNE_PLANCK_HPP_

// Jaybenne includes
#include "jaybenne.hpp"

namespace jaybenne {

//----------------------------------------------------------------------------------------
//! \fn Real sample_Planck_energy
//! \brief Efficiently samples the Planck distribution for particle energy
//!        rng_gen: RNG pool
//!        T: distribution temperature
KOKKOS_FORCEINLINE_FUNCTION
Real sample_Planck_energy(RngGen &rng_gen, const Real &sb, const Real &temp) {
  // Sampling method from Everett & Cashwell 1972
  const Real xi0 = rng_gen.drand();
  const Real rhs = xi0 * std::pow(M_PI, 4.0) / 90.0;
  int l = 1;
  Real ll = 1.0;
  while (l < 100) {
    Real lhs = 0.0;
    for (int j = 1; j <= l; j++) {
      lhs += std::pow(static_cast<Real>(j), -4.0);
    }
    if (lhs >= rhs) {
      ll = static_cast<Real>(l);
      break;
    }
    l++;
  }

  const Real xi1 = rng_gen.drand();
  const Real xi2 = rng_gen.drand();
  const Real xi3 = rng_gen.drand();
  const Real xi4 = rng_gen.drand();
  return -(1.0 / ll) * std::log(xi1 * xi2 * xi3 * xi4) * sb * temp;
}

} // namespace jaybenne

#endif // JAYBENNE_PLANCK_HPP_
