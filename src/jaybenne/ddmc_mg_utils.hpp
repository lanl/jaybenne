#ifndef JAYBENNE_DDMC_MG_UTILS_HPP_
#define JAYBENNE_DDMC_MG_UTILS_HPP_

// Host configuration file
#include "planck.hpp"

// ddmc_mg_utils
namespace jaybenne {

//----------------------------------------------------------------------------------------
// helper struct to encapsulate data needed to integrate DDMC probabilities over groups
struct ddmc_mg_leak_args {
  // const int &gmode;     // average type to use in MG opacity (usually Rosseland)
  const int &n_nubins;  // number of frequency groups (or bins)
  const Real &dlnu;     // log-spacing of frequency groups
  const Real &hd;       // Planck constant
  const Real &kboltd;   // Boltzmann constant
  const Real &tau_ddmc; // DDMC cell optical-thickness threshold
  const Real &dx_lmin;  // min cell length scale used to activate DDMC
  const Real &dx_umin;  // min cell length scale used to activate DDMC
  const Real &dx_l;     // lower cell length
  const Real &dx_u;     // upper cell length
  const Real &rho_l;    // lower cell density
  const Real &rho_u;    // upper cell density
  const Real &temp_l;   // lower cell temperature
  const Real &temp_u;   // upper cell temperature
};

// helper struct for in-cell MG DDMC processes
struct ddmc_mg_cell_args {
  const int &n_nubins;  // number of frequency groups (or bins)
  const Real &dlnu;     // log-spacing of frequency groups
  const Real &hd;       // Planck constant
  const Real &kboltd;   // Boltzmann constant
  const Real &tau_ddmc; // DDMC cell optical-thickness threshold
  const Real &dx_min;   // min cell length scale used to activate DDMC
  const Real &rho;      // cell density
  const Real &temp;     // cell temperature
};

//----------------------------------------------------------------------------------------
template <typename OP, typename SC>
KOKKOS_FORCEINLINE_FUNCTION std::pair<Real, Real>
calc_ddmc_mg_leak_numdenom(const OP &abs, const SC &sct, const ddmc_mg_leak_args &dmg,
                           const ParArray1D<Real> &nu_bins, const bool &use_lo) {

  // define extrapolation distance (Habetler & Matkowski 1975)
  constexpr Real lam_ext = 0.7104;

  // evaluate a face temperature for Planck evaluation
  const Real temp_f = std::max(dmg.temp_l, dmg.temp_u);

  // initialize unnnormalized Planck integral and probability
  Real planck_sum = 0.0;
  Real leak_sum = 0.0;
  int nnubins_used = 0;

  // integrate
  for (int n = 0; n < dmg.n_nubins; ++n) {

    // evaluate face opacities at nu_bins(n)
    const Real ss_l = sct.ScatteringCoefficient(dmg.rho_l, dmg.temp_l, n);
    const Real aa_l = abs.AbsorptionCoefficient(dmg.rho_l, dmg.temp_l, n);
    const Real ss_u = sct.ScatteringCoefficient(dmg.rho_u, dmg.temp_u, n);
    const Real aa_u = abs.AbsorptionCoefficient(dmg.rho_u, dmg.temp_u, n);

    // calculate optical thicknesses from lower and upper cell
    const Real tau_lmin = dmg.dx_lmin * (ss_l + aa_l);
    const Real tau_umin = dmg.dx_umin * (ss_u + aa_u);
    const Real tau_l = dmg.dx_l * (ss_l + aa_l);
    const Real tau_u = dmg.dx_u * (ss_u + aa_u);

    // check if group is included
    const bool use_grp = (use_lo ? tau_lmin > dmg.tau_ddmc : tau_umin > dmg.tau_ddmc);

    if (use_grp) {
      nnubins_used++;

      const Real mtau_l = tau_lmin > dmg.tau_ddmc ? tau_l : 2.0 * lam_ext;
      const Real mtau_u = tau_umin > dmg.tau_ddmc ? tau_u : 2.0 * lam_ext;

      // calculate per-nu-bin leakage
      const Real Pg = 2.0 / (3.0 * (mtau_l + mtau_u));

      // convert to energy units for Planck integral
      const Real ee = dmg.hd * nu_bins(n);
      const Real dee = ee * dmg.dlnu;

      // get an unnormalized, non-dimensional Planck integral over group
      const Real bg = midpoint_Planck(dmg.kboltd * temp_f, ee, dee);

      // aggregate values
      planck_sum += bg;
      leak_sum += bg * Pg;
    }
  }

  PARTHENON_REQUIRE(nnubins_used ? planck_sum > 0.0 : true, "Planck integral <= 0.0");
  return {leak_sum, planck_sum};
}

//----------------------------------------------------------------------------------------
// integrate unnormalized leakage probability on high or low side of face
template <typename OP, typename SC>
KOKKOS_FORCEINLINE_FUNCTION Real calc_ddmc_mg_leakprob(const OP &abs, const SC &sct,
                                                       const ddmc_mg_leak_args &dmg,
                                                       const ParArray1D<Real> &nu_bins,
                                                       const bool &use_lo) {
  const auto leak_sum_pair = calc_ddmc_mg_leak_numdenom(abs, sct, dmg, nu_bins, use_lo);
  if (!(leak_sum_pair.second > 0.0)) return 0.0;
  // normalize sum (so that leakage probability is an average)
  return leak_sum_pair.first / leak_sum_pair.second;
}

//----------------------------------------------------------------------------------------
template <typename OP, typename SC>
KOKKOS_FORCEINLINE_FUNCTION int
sample_leakage_group(const OP &abs, const SC &sct, const ddmc_mg_leak_args &dmg,
                     const ParArray1D<Real> &nu_bins, const bool &use_lo,
                     RngGen &rng_gen) {

  // first get CDF totals
  const auto leak_sum_pair = calc_ddmc_mg_leak_numdenom(abs, sct, dmg, nu_bins, use_lo);
  const Real &leak_tot_sum = leak_sum_pair.first;
  const Real &planck_tot_sum = leak_sum_pair.second;

  // define extrapolation distance (Habetler & Matkowski 1975)
  constexpr Real lam_ext = 0.7104;

  // evaluate a face temperature for Planck evaluation
  const Real temp_f = std::max(dmg.temp_l, dmg.temp_u);

  Real leak_sum = 0.0;
  Real planck_sum = 0.0;

  // sample
  const Real rand1 = leak_tot_sum * rng_gen.drand();
  int inu_sampled = -1; // poisoned initialization

  // integrate
  for (int n = 0; n < dmg.n_nubins; ++n) {

    // evaluate face opacities at nu_bins(n)
    const Real ss_l = sct.ScatteringCoefficient(dmg.rho_l, dmg.temp_l, n);
    const Real aa_l = abs.AbsorptionCoefficient(dmg.rho_l, dmg.temp_l, n);
    const Real ss_u = sct.ScatteringCoefficient(dmg.rho_u, dmg.temp_u, n);
    const Real aa_u = abs.AbsorptionCoefficient(dmg.rho_u, dmg.temp_u, n);

    // calculate optical thicknesses from lower and upper cell
    const Real tau_lmin = dmg.dx_lmin * (ss_l + aa_l);
    const Real tau_umin = dmg.dx_umin * (ss_u + aa_u);
    const Real tau_l = dmg.dx_l * (ss_l + aa_l);
    const Real tau_u = dmg.dx_u * (ss_u + aa_u);

    // check if group is included
    const bool use_grp = (use_lo ? tau_lmin > dmg.tau_ddmc : tau_umin > dmg.tau_ddmc);

    if (use_grp) {
      const Real mtau_l = tau_lmin > dmg.tau_ddmc ? tau_l : 2.0 * lam_ext;
      const Real mtau_u = tau_umin > dmg.tau_ddmc ? tau_u : 2.0 * lam_ext;

      // calculate per-nu-bin leakage
      const Real Pg = 2.0 / (3.0 * (mtau_l + mtau_u));

      // convert to energy units for Planck integral
      const Real ee = dmg.hd * nu_bins(n);
      const Real dee = ee * dmg.dlnu;

      // get an unnormalized, non-dimensional Planck integral over group
      const Real bg = midpoint_Planck(dmg.kboltd * temp_f, ee, dee);

      // aggregate values
      planck_sum += bg;
      leak_sum += bg * Pg;
      if (leak_sum > rand1) {
        inu_sampled = n;
        break;
      }
    }
  }

  PARTHENON_DEBUG_REQUIRE(inu_sampled >= 0, "inu_sampled < 0");

  // return sampled energy value (in units of energy)
  return inu_sampled;
}

//----------------------------------------------------------------------------------------
// calculate: absorption, outscatter probability
template <typename OP, typename SC>
KOKKOS_FORCEINLINE_FUNCTION std::pair<Real, Real>
calc_ddmc_mg_probs(const OP &abs, const SC &sct, const ddmc_mg_cell_args &dmgc,
                   const ParArray1D<Real> &nu_bins) {

  // initialize unnnormalized Planck integral and probability
  Real planck_sum = 0.0;
  Real abs_sum = 0.0;
  Real abs_tot_sum = 0.0;

  // integrate
  for (int n = 0; n < dmgc.n_nubins; ++n) {

    // evaluate face opacities at nu_bins(n)
    const Real ss = sct.ScatteringCoefficient(dmgc.rho, dmgc.temp, n);
    const Real aa = abs.AbsorptionCoefficient(dmgc.rho, dmgc.temp, n);

    // convert to energy units for Planck integral
    const Real ee = dmgc.hd * nu_bins(n);
    const Real dee = ee * dmgc.dlnu;

    // get an unnormalized, non-dimensional Planck integral over group
    const Real bg = midpoint_Planck(dmgc.kboltd * dmgc.temp, ee, dee);

    // sum total (Planck)
    abs_tot_sum += bg * aa;

    // check group inclusion
    if (dmgc.dx_min * (ss + aa) > dmgc.tau_ddmc) {

      // aggregate values
      planck_sum += bg;
      abs_sum += bg * aa;
    }
  }

  PARTHENON_DEBUG_REQUIRE(planck_sum > 0.0, "planck_sum = 0: no DDMC groups in DDMC.");

  // 1st entry = DDMC absorption, 2nd = out-scatter probability
  if (abs_tot_sum > 0.0) {
    return {abs_sum / planck_sum, 1.0 - abs_sum / abs_tot_sum};
  } else {
    // no possible outscatter if there is no absorption/redistribution
    return {abs_sum / planck_sum, 0.0};
  }
}

//----------------------------------------------------------------------------------------
// sample out-scatter IMC group
// NOTE(MGDDMC): this routine is assuming Kirchhoff's Law for emissivity (LTE)
template <typename OP, typename SC>
KOKKOS_FORCEINLINE_FUNCTION int
sample_ddmc2imc_outscatter(const OP &abs, const SC &sct, const ddmc_mg_cell_args &dmgc,
                           const ParArray1D<Real> &nu_bins, RngGen &rng_gen,
                           const bool stay_in = false) {

  Real scat_out_tot_sum = 0.0;

  // integrate total cdf value
  for (int n = 0; n < dmgc.n_nubins; ++n) {

    // evaluate face opacities at nu_bins(n)
    const Real ss = sct.ScatteringCoefficient(dmgc.rho, dmgc.temp, n);
    const Real aa = abs.AbsorptionCoefficient(dmgc.rho, dmgc.temp, n);

    // check group exclusion
    const bool is_ddmc_grp = dmgc.dx_min * (ss + aa) > dmgc.tau_ddmc;
    if (stay_in ? is_ddmc_grp : !is_ddmc_grp) {
      // get an unnormalized, non-dimensional Planck integral over group
      const Real ee = dmgc.hd * nu_bins(n);
      const Real dee = ee * dmgc.dlnu;
      const Real bg = midpoint_Planck(dmgc.kboltd * dmgc.temp, ee, dee);
      scat_out_tot_sum += bg * aa;
    }
  }

  // sample
  const Real rand1 = scat_out_tot_sum * rng_gen.drand();
  int inu_sampled = -1; // poisoned initialization

  Real abs_sum = 0.0;

  // find bin for sample
  for (int n = 0; n < dmgc.n_nubins; ++n) {

    // evaluate face opacities at nu_bins(n)
    const Real ss = sct.ScatteringCoefficient(dmgc.rho, dmgc.temp, n);
    const Real aa = abs.AbsorptionCoefficient(dmgc.rho, dmgc.temp, n);

    // check group exclusion
    const bool is_ddmc_grp = dmgc.dx_min * (ss + aa) > dmgc.tau_ddmc;
    if (stay_in ? is_ddmc_grp : !is_ddmc_grp) {

      // convert to energy units for Planck integral
      const Real ee = dmgc.hd * nu_bins(n);
      const Real dee = ee * dmgc.dlnu;

      // get an unnormalized, non-dimensional Planck integral over group
      const Real bg = midpoint_Planck(dmgc.kboltd * dmgc.temp, ee, dee);

      // aggregate values
      abs_sum += bg * aa;

      if (abs_sum > rand1) {
        inu_sampled = n;
        break;
      }
    }
  }

  PARTHENON_DEBUG_REQUIRE(inu_sampled >= 0, "inu_sampled < 0");
  return inu_sampled;
}

} // namespace jaybenne

#endif // JAYBENNE_DDMC_MG_UTILS_HPP_
