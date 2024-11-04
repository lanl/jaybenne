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
#ifndef JAYBENNE_SCATTERING_HPP_
#define JAYBENNE_SCATTERING_HPP_

namespace jaybenne {

//----------------------------------------------------------------------------------------
//! \fn  void scatter
//! \brief TODO(RTW): template on ScatteringModel, when pertinent
KOKKOS_FORCEINLINE_FUNCTION
void scatter(RngGen &rng_gen, const Real &vv, Real &vx_out, Real &vy_out, Real &vz_out) {
  const Real mu = 2.0 * rng_gen.drand() - 1.0;
  const Real phi = 2.0 * M_PI * rng_gen.drand();
  const Real stheta = std::sqrt(1.0 - mu * mu);
  vx_out = vv * stheta * std::cos(phi);
  vy_out = vv * stheta * std::sin(phi);
  vz_out = vv * mu;
}

} // namespace jaybenne

#endif // JAYBENNE_SCATTERING_HPP_
