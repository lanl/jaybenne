//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
// scattering input data helper structs
struct cell_scat_args {
  const int &b;   // block index
  const int &ip;  // x/X1 block cell index
  const int &jp;  // y/X2 block cell index
  const int &kp;  // z/X3 block cell index
  const Real &ff; // Fleck factor
  const Real &aa; // absorption opacity (1/length)
  const Real &ss; // scattering opacity (1/length)
  // frequency data
  const int &n_nubinsd; // number of frequency groups
  const Real &hd;       // Planck constant
};
struct ptcl_scat_args {
  RngGen &rng_gen;
  const Real &vv;
  Real &vx; // particle x/X1-direction speed
  Real &vy; // particle y/X2-direction speed
  Real &vz; // particle z/X3-direction speed
  int &inu; // particle group
};

//----------------------------------------------------------------------------------------
//! \fn  void isotropic direction redistribution
KOKKOS_FORCEINLINE_FUNCTION
void sample_vol_iso_dir(ptcl_scat_args sa) {
  const Real mu = 2.0 * sa.rng_gen.drand() - 1.0;
  const Real phi = 2.0 * M_PI * sa.rng_gen.drand();
  const Real stheta = std::sqrt(1.0 - mu * mu);
  sa.vx = sa.vv * stheta * std::cos(phi);
  sa.vy = sa.vv * stheta * std::sin(phi);
  sa.vz = sa.vv * mu;
}

//----------------------------------------------------------------------------------------
//! \fn  void scatter kernel
//! TODO(BRR): template on scattering kernel type?
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION void scatter_kernel(const T &vmesh, cell_scat_args csa,
                                                const ParArray1D<Real> &nu_bins,
                                                ptcl_scat_args psa) {
  namespace fj = field::jaybenne;

  // sample direction isotropically (see TODO above about more general kernels)
  sample_vol_iso_dir(psa);

  // sample whether effective scattering occurred
  const Real rand1 = psa.rng_gen.drand();
  if (rand1 * ((1.0 - csa.ff) * csa.aa + csa.ss) < (1.0 - csa.ff) * csa.aa) {

    // Sample energy from CDF
    const Real rand2 = psa.rng_gen.drand();
    int n;
    for (n = 0; n < csa.n_nubinsd; n++) {
      if (vmesh(csa.b, fj::emission_cdf(n), csa.kp, csa.jp, csa.ip) >= rand2) {
        break;
      }
    }

    // reset particle frequency
    psa.inu = n;
  }
}

} // namespace jaybenne

#endif // JAYBENNE_SCATTERING_HPP_
