//========================================================================================
// (C) (or copyright) 2024-2025. Triad National Security, LLC. All rights reserved.
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
#ifndef JAYBENNE_TRANSPORT_UTILS_HPP_
#define JAYBENNE_TRANSPORT_UTILS_HPP_

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace jaybenne {

// position displacements into next cell, for IMC and DDMC
// TODO(@pdmllen): how to best set prefactor?
KOKKOS_FORCEINLINE_FUNCTION
constexpr Real eps_imc_offset() { return 1.0e6 * parthenon::robust::EPS(); }

KOKKOS_FORCEINLINE_FUNCTION
constexpr Real eps_ddmc_offset() { return 1.0e8 * parthenon::robust::EPS(); }

KOKKOS_FORCEINLINE_FUNCTION
void sample_face_iso_dir(const Real vv, RngGen &rng_gen, Real &v1, Real &v2, Real &v3) {

  // sample principle direction and azimuthal angle about principle direction
  const Real mu = std::sqrt(rng_gen.drand());
  const Real nu = std::sqrt(1.0 - mu * mu);
  const Real phi = 2.0 * M_PI * rng_gen.drand();

  // v1 is principle velocity component
  v1 = vv * mu;
  v2 = vv * nu * std::cos(phi);
  v3 = vv * nu * std::sin(phi);
}

// helper struct to encapsulate transport step arguments
struct tran_step_args {
  // Constants
  RngGen &rng_gen;     // ref to random number generator
  const Real &t_start; // sim time at beginning of time step
  const Real &dt;      // sim time step size/duration
  const Real &ff;      // Fleck factor
  const Real &aa;      // absorption opacity (1/length)
  const Real &ss;      // scattering opacity (1/length)
  const Real &vv;      // particle speed (should be c)
  Real &vx;            // particle x/X1-direction speed
  Real &vy;            // particle y/X2-direction speed
  Real &vz;            // particle z/X3-direction speed
  const Real &dx_push; // minimum spatial cell dimension
  const bool &multi_d; // 2D or 3D
  const bool &three_d; // 3D
  const Real &xl;      // min-x bound (at face) of cell
  const Real &yl;      // min-y bound (at face) of cell
  const Real &zl;      // min-z bound (at face) of cell
  const Real &xu;      // max-x bound (at face) of cell
  const Real &yu;      // max-y bound (at face) of cell
  const Real &zu;      // max-z bound (at face) of cell
  // Updated during push
  Real &t;            // particle time
  Real &x;            // particle x/X1-coordinate
  Real &y;            // particle y/X2-coordinate
  Real &z;            // particle z/X3-coordinate
  Real &ww;           // energy-weight of the particle
  Real &fraction;     // weight/initial weight of the particle
  Real &e_abs;        // energy-weight absorbed in this step
  bool &is_scattered; // indicator for scattering in the step
  bool &is_census;    // indicator for end of census
  bool &is_absorbed;  // indicator for analog absorption
};

// helper struct to encapsulate DDMC step arguments
struct ddmc_step_args {
  // Constants
  RngGen &rng_gen;     // reference to random number generator
  const Real &t_start; // sim time at beginning of time step
  const Real &dt;      // sim time step size/duration
  const Real &ff;      // Fleck factor
  const Real &aa;      // absorption opacity (1/length)
  const Real &ss;      // scattering opacity (1/length)
  const Real &vv;      // particle speed (should be c)
  const bool &multi_d; // 2D or 3D
  const bool &three_d; // 3D
  const Real &xl;      // min-x bound (at face) of cell
  const Real &yl;      // min-y bound (at face) of cell
  const Real &zl;      // min-z bound (at face) of cell
  const Real &xu;      // max-x bound (at face) of cell
  const Real &yu;      // max-y bound (at face) of cell
  const Real &zu;      // max-z bound (at face) of cell
  const Real &Px_l;    // leakage probability for min-x cell face
  const Real &Py_l;    // leakage probability for min-y cell face
  const Real &Pz_l;    // leakage probability for min-z cell face
  const Real &Px_u;    // leakage probability for max-x cell face
  const Real &Py_u;    // leakage probability for max-y cell face
  const Real &Pz_u;    // leakage probability for max-z cell face
  // Updated during push
  Real &t;            // particle time
  Real &x;            // particle x/X1-coordinate
  Real &y;            // particle y/X2-coordinate
  Real &z;            // particle z/X3-coordinate
  Real &vx;           // particle x/X1-direction speed
  Real &vy;           // particle y/X2-direction speed
  Real &vz;           // particle z/X3-direction speed
  int &ip;            // x/X1 block index
  int &jp;            // y/X2 block index
  int &kp;            // z/X3 block index
  bool &is_absorbed;  // indicator for absorption in the step
  bool &is_scattered; // indicator for scattering in the step
  bool &is_census;    // indicator for end of census
};

KOKKOS_FORCEINLINE_FUNCTION
void ptcl_transport_step(tran_step_args tra, const double cutoff) {

  // use distances and convert to time after min is determined to reduce division ops
  enum Event { BOUND, COLLISION, CENSUS };

  constexpr Real rmin = std::numeric_limits<Real>::min();
  constexpr Real rmax = std::numeric_limits<Real>::max();

  // scatter
  const Real sigma_scatter = tra.ss + (1.0 - tra.ff) * tra.aa;
  const Real sigma_collision =
      sigma_scatter + ((tra.fraction < cutoff) ? tra.ff * tra.aa : 0.0);
  const Real lam_collision = 1.0 / (sigma_collision + rmin);
  const Real dx_collision = -lam_collision * std::log(tra.rng_gen.drand());

  // census
  const Real dx_end = tra.vv * ((tra.t_start + tra.dt) - tra.t);

  int face = -1;
  Real dx_bound = rmax;
  Real dx_push_x = (tra.vx > 0.0) ? tra.vv * (tra.xu - tra.x) / tra.vx
                                  : tra.vv * (tra.xl - tra.x) / tra.vx;
  Real dx_push_y = rmax;
  if (tra.multi_d)
    dx_push_y = (tra.vy > 0.0) ? tra.vv * (tra.yu - tra.y) / tra.vy
                               : tra.vv * (tra.yl - tra.y) / tra.vy;

  Real dx_push_z = rmax;
  if (tra.three_d)
    dx_push_z = (tra.vz > 0.0) ? tra.vv * (tra.zu - tra.z) / tra.vz
                               : tra.vv * (tra.zl - tra.z) / tra.vz;

  // default to x direction
  dx_bound = dx_push_x;
  face = (tra.vx > 0.0) ? 1 : 0;
  // y check
  if ((dx_push_y < dx_push_x) && (dx_push_y < dx_push_z)) {
    dx_bound = dx_push_y;
    face = (tra.vy > 0.0) ? 3 : 2;
  }
  // z check
  if ((dx_push_z < dx_push_x) && (dx_push_z < dx_push_y)) {
    dx_bound = dx_push_z;
    face = (tra.vz > 0.0) ? 5 : 4;
  }

  // default to scatter
  Real dx_push = dx_collision;
  Event event = COLLISION;
  tra.is_scattered = true;
  tra.is_census = false;

  // check for census
  if (dx_end < dx_push) {
    dx_push = dx_end;
    event = CENSUS;
    tra.is_scattered = false;
    tra.is_census = true;
  }
  // check for bound
  if (dx_bound < dx_push) {
    dx_push = dx_bound;
    event = BOUND;
    tra.is_scattered = false;
    tra.is_census = false;
  }

  // set distance to translate particle position
  const Real dt_push = dx_push / tra.vv;

  // push
  tra.t += dt_push;
  tra.x += tra.vx * dt_push;
  tra.y += tra.multi_d * tra.vy * dt_push;
  tra.z += tra.three_d * tra.vz * dt_push;

  if (tra.fraction < cutoff) {
    if (event == COLLISION && tra.rng_gen.drand() > sigma_scatter / sigma_collision) {
      // post processing uses is_absorbed to deposit full particle energy into matieral
      tra.is_absorbed = true;
      tra.e_abs = 0.0;
      tra.is_scattered = false;
    }
  } else {
    // attenuate particle
    const Real exp_factor = exp(-dx_push * tra.ff * tra.aa);
    tra.e_abs = tra.ww * (1.0 - exp_factor);
    tra.ww = tra.ww - tra.e_abs;
    tra.fraction = tra.fraction * exp_factor;
  }

  // push time slightly into next time step if particle is at end of time step (census)
  const bool is_at_end =
      fuzzy_equal(tra.t, tra.t_start + tra.dt, tra.dt, eps_imc_offset());
  tra.t = is_at_end ? tra.t_start + (1.0 + eps_imc_offset()) * tra.dt : tra.t;

  // handle face crossings by snapping to face and adding epsilon offset
  if (event == BOUND) {
    if (face == 0 || face == 1) {
      const Real offset = eps_imc_offset() * (tra.xu - tra.xl);
      tra.x = (face == 0) ? (tra.xl - offset) : tra.xu + offset;
    } else if (face == 2 || face == 3) {
      const Real offset = eps_imc_offset() * (tra.yu - tra.yl);
      tra.y = (face == 2) ? (tra.yl - offset) : tra.yu + offset;
    } else { // (face == 4 || face == 5) {
      const Real offset = eps_imc_offset() * (tra.zu - tra.zl);
      tra.z = (face == 4) ? tra.zl - offset : tra.zu + offset;
    }
  }
}

// TODO(RTW): add effective out-scattering from DDMC when multigroup is enabled
KOKKOS_FORCEINLINE_FUNCTION
void ptcl_ddmc_step(ddmc_step_args dia) {

  const Real rmin = std::numeric_limits<Real>::min();

  // calculate cell dimensions
  const Real eps = eps_ddmc_offset(); // move particles eps_ddmc_offset() into next cell
  const Real dx = dia.xu - dia.xl;
  const Real dy = dia.yu - dia.yl;
  const Real dz = dia.zu - dia.zl;

  // calculate leakage opacity from face probabilites
  const Real leakx_l = dia.Px_l / dx;
  const Real leakx_u = dia.Px_u / dx;
  const Real leaky_l = dia.Py_l / dy;
  const Real leaky_u = dia.Py_u / dy;
  const Real leakz_l = dia.Pz_l / dz;
  const Real leakz_u = dia.Pz_u / dz;
  const Real leak_tot = leakx_l + leakx_u + leaky_l + leaky_u + leakz_l + leakz_u;

  // calculate time to DDMC event and compare to time to end of time step (census)
  const Real cdf_ddmc = dia.ff * dia.aa + leak_tot + rmin;
  const Real dt_ddmc = -std::log(dia.rng_gen.drand()) / (dia.vv * cdf_ddmc);
  const Real dt_end = (dia.t_start + dia.dt) - dia.t;
  const bool is_ddmc_event = dt_ddmc < dt_end;

  // update particle time
  const Real dt_push = std::min(dt_ddmc, dt_end);
  dia.t += dt_push;

  if (is_ddmc_event) {

    // sample DDMC CDF
    const Real xi = cdf_ddmc * dia.rng_gen.drand();

    if (xi < dia.ff * dia.aa) {

      // particle will be absorbed
      dia.is_absorbed = true;

    } else if (xi < dia.ff * dia.aa + leak_tot) {

      // TODO(RTW): only sample direction if adjacent cell is below tau_ddmc

      // particle will leak to an adjacent cell
      const Real xim = xi - dia.ff * dia.aa;
      if (xim < leakx_l) {
        // leak in negative x/X1 direction
        dia.ip -= 1;
        // update position
        dia.x = dia.xl - eps * dx;
        dia.y = dia.yl + 0.5 * dy;
        dia.z = dia.zl + 0.5 * dz;
        // sample direction
        sample_face_iso_dir(-dia.vv, dia.rng_gen, dia.vx, dia.vy, dia.vz);
      } else if (xim < leakx_l + leakx_u) {
        // leak in positive x/X1 direction
        dia.ip += 1;
        // update position
        dia.x = dia.xu + eps * dx;
        dia.y = dia.yl + 0.5 * dy;
        dia.z = dia.zl + 0.5 * dz;
        // sample direction
        sample_face_iso_dir(dia.vv, dia.rng_gen, dia.vx, dia.vy, dia.vz);
      } else if (xim < leakx_l + leakx_u + leaky_l) {
        // leak in negative y/X2 direction
        dia.jp -= dia.multi_d;
        // update position
        dia.y = dia.yl - eps * dy;
        dia.z = dia.zl + 0.5 * dz;
        dia.x = dia.xl + 0.5 * dx;
        // sample direction
        sample_face_iso_dir(-dia.vv, dia.rng_gen, dia.vy, dia.vz, dia.vx);
      } else if (xim < leakx_l + leakx_u + leaky_l + leaky_u) {
        // leak in positive y/X2 direction
        dia.jp += dia.multi_d;
        // update position
        dia.y = dia.yu + eps * dy;
        dia.z = dia.zl + 0.5 * dz;
        dia.x = dia.xl + 0.5 * dx;
        // sample direction
        sample_face_iso_dir(dia.vv, dia.rng_gen, dia.vy, dia.vz, dia.vx);
      } else if (xim < leakx_l + leakx_u + leaky_l + leaky_u + leakz_l) {
        // leak in negative z/X3 direction
        dia.kp -= dia.three_d;
        // update position
        dia.z = dia.zl - eps * dz;
        dia.x = dia.xl + 0.5 * dx;
        dia.y = dia.yl + 0.5 * dy;
        // sample direction
        sample_face_iso_dir(-dia.vv, dia.rng_gen, dia.vz, dia.vx, dia.vy);
      } else if (xim <= leak_tot) {
        // leak in positive z/X3 direction
        dia.kp += dia.three_d;
        // update position
        dia.z = dia.zu + eps * dz;
        dia.x = dia.xl + 0.5 * dx;
        dia.y = dia.yl + 0.5 * dy;
        // sample direction
        sample_face_iso_dir(dia.vv, dia.rng_gen, dia.vz, dia.vx, dia.vy);
      }
    }
  } else {
    // census particle: resample pos and dir (it may be transported next time step)
    dia.z = dia.zl + dia.rng_gen.drand() * dz;
    dia.x = dia.xl + dia.rng_gen.drand() * dx;
    dia.y = dia.yl + dia.rng_gen.drand() * dy;
    const Real mu = 1.0 - 2.0 * dia.rng_gen.drand();
    const Real nu = std::sqrt(1.0 - mu * mu);
    const Real phi = 2.0 * M_PI * dia.rng_gen.drand();
    dia.vz = dia.vv * mu;
    dia.vx = dia.vv * nu * std::cos(phi);
    dia.vy = dia.vv * nu * std::sin(phi);

    // ensure time is slightly past census time (TODO: use eps_ddmc_offset?)
    dia.t = dia.t_start + (1.0 + eps_imc_offset()) * dia.dt;
    dia.is_census = true;
  }
}

KOKKOS_FORCEINLINE_FUNCTION
void ptcl_ddmc_albedo(ddmc_step_args dia, bool &is_rejected) {

  constexpr Real lam_ext = 0.7104;
  const Real dx = dia.xu - dia.xl;
  const Real dy = dia.yu - dia.yl;
  const Real dz = dia.zu - dia.zl;

  // permit into DDMC automatically if originating from census
  if (fuzzy_equal(dia.t, dia.t_start, dia.dt, 2.0 * eps_imc_offset())) {
    dia.x = 0.5 * (dia.xl + dia.xu);
    dia.y = 0.5 * (dia.yl + dia.yu);
    dia.z = 0.5 * (dia.zl + dia.zu);
    is_rejected = false;
    return;
  }

  // IMC->DDMC entry conditions at each cell face
  const bool vxp = (dia.vx > 0.0);
  const bool vyp = (dia.vy > 0.0);
  const bool vzp = (dia.vz > 0.0);
  const Real w_eps_imc_offset = 2.5 * eps_imc_offset();
  const bool lowx = (vxp && fuzzy_equal(dia.x, dia.xl, dx, w_eps_imc_offset));
  const bool higx = (!vxp && fuzzy_equal(dia.x, dia.xu, dx, w_eps_imc_offset));
  const bool lowy = (vyp && fuzzy_equal(dia.y, dia.yl, dy, w_eps_imc_offset));
  const bool higy = (!vyp && fuzzy_equal(dia.y, dia.yu, dy, w_eps_imc_offset));
  const bool lowz = (vzp && fuzzy_equal(dia.z, dia.zl, dz, w_eps_imc_offset));
  const bool higz = (!vzp && fuzzy_equal(dia.z, dia.zu, dz, w_eps_imc_offset));

  // check that coordinate is at cell edge (only possible coming from IMC)
  if (lowx) {
    // lower x-face

    // vx should be non-negative
    const Real Px_l = (2.0 / 3.0) / ((dia.aa + dia.ss) * dx + 2.0 * lam_ext);
    const Real P = 2.0 * Px_l * (1.0 + 1.5 * dia.vx / dia.vv);

    // sample lower x-albedo
    if (dia.rng_gen.drand() > P) {
      // sample direction
      sample_face_iso_dir(-dia.vv, dia.rng_gen, dia.vx, dia.vy, dia.vz);
      // set particle z position slightly above face
      dia.x = dia.xl - eps_ddmc_offset() * dx;
      // set rejection indicator
      is_rejected = true;
    }

  } else if (higx) {
    // upper x-face

    // vx should be non-positive
    const Real Px_u = (2.0 / 3.0) / ((dia.aa + dia.ss) * dx + 2.0 * lam_ext);
    const Real P = 2.0 * Px_u * (1.0 - 1.5 * dia.vx / dia.vv);

    // sample upper x-albedo
    if (dia.rng_gen.drand() > P) {
      // sample direction
      sample_face_iso_dir(dia.vv, dia.rng_gen, dia.vx, dia.vy, dia.vz);
      // set particle z position slightly above face
      dia.x = dia.xu + eps_ddmc_offset() * dx;
      // set rejection indicator
      is_rejected = true;
    }

  } else if (lowy && dia.multi_d) {
    // lower y-face

    // vy should be non-negative
    const Real Py_l = (2.0 / 3.0) / ((dia.aa + dia.ss) * dy + 2.0 * lam_ext);
    const Real P = 2.0 * Py_l * (1.0 + 1.5 * dia.vy / dia.vv);

    // sample lower y-albedo
    if (dia.rng_gen.drand() > P) {
      // sample direction
      sample_face_iso_dir(-dia.vv, dia.rng_gen, dia.vy, dia.vz, dia.vx);
      // set particle y position slightly above face
      dia.y = dia.yl - eps_ddmc_offset() * dy;
      // set rejection indicator
      is_rejected = true;
    }

  } else if (higy && dia.multi_d) {
    // upper y-face

    // vy should be non-positive
    const Real Py_u = (2.0 / 3.0) / ((dia.aa + dia.ss) * dy + 2.0 * lam_ext);
    const Real P = 2.0 * Py_u * (1.0 - 1.5 * dia.vy / dia.vv);

    // sample upper y-albedo
    if (dia.rng_gen.drand() > P) {
      // sample direction
      sample_face_iso_dir(dia.vv, dia.rng_gen, dia.vy, dia.vz, dia.vx);
      // set particle y position slightly above face
      dia.y = dia.yu + eps_ddmc_offset() * dy;
      // set rejection indicator
      is_rejected = true;
    }

  } else if (lowz && dia.three_d) {
    // lower z-face

    // vz should be non-negative
    const Real Pz_l = (2.0 / 3.0) / ((dia.aa + dia.ss) * dz + 2.0 * lam_ext);
    const Real P = 2.0 * Pz_l * (1.0 + 1.5 * dia.vz / dia.vv);

    // sample lower z-albedo
    if (dia.rng_gen.drand() > P) {
      // sample direction
      sample_face_iso_dir(-dia.vv, dia.rng_gen, dia.vz, dia.vx, dia.vy);
      // set particle z position slightly above face
      dia.z = dia.zl - eps_ddmc_offset() * dz;
      // set rejection indicator
      is_rejected = true;
    }

  } else if (higz && dia.three_d) {
    // upper z-face

    // vz should be non-positive
    const Real Pz_u = (2.0 / 3.0) / ((dia.aa + dia.ss) * dz + 2.0 * lam_ext);
    const Real P = 2.0 * Pz_u * (1.0 - 1.5 * dia.vz / dia.vv);

    // sample upper y-albedo
    if (dia.rng_gen.drand() > P) {
      // sample direction
      sample_face_iso_dir(dia.vv, dia.rng_gen, dia.vz, dia.vx, dia.vy);
      // set particle z position slightly above face
      dia.z = dia.zu + eps_ddmc_offset() * dz;
      // set rejection indicator
      is_rejected = true;
    }
  }

  // if admitted, set to the cell center (to avoid redoing this in next step)
  if (!is_rejected) {
    dia.x = 0.5 * (dia.xl + dia.xu);
    dia.y = 0.5 * (dia.yl + dia.yu);
    dia.z = 0.5 * (dia.zl + dia.zu);
  }
}

KOKKOS_FORCEINLINE_FUNCTION
void ptcl_ddmc_to_imc(tran_step_args tra) {

  // set tolerance for checking particle coordinate
  constexpr Real eps = parthenon::robust::EPS();

  // cell dimensions
  const Real dx = tra.xu - tra.xl;
  const Real dy = tra.yu - tra.yl;
  const Real dz = tra.zu - tra.zl;

  // two possibilities here
  bool res_valid = false;

  for (int r = 0; r < 2; ++r) {

    const Real resf = 1.0 / static_cast<Real>(r + 1);

    // sample spatial coordinate and velocity
    if (fuzzy_equal(tra.x, tra.xl + resf * eps_ddmc_offset() * dx, dx, eps)) {
      // sample y and z
      tra.y = tra.yl + tra.rng_gen.drand() * resf * dy;
      tra.z = tra.zl + tra.rng_gen.drand() * resf * dz;
      // sample direction
      sample_face_iso_dir(tra.vv, tra.rng_gen, tra.vx, tra.vy, tra.vz);
      // break loop
      res_valid = true;
      break;
    } else if (fuzzy_equal(tra.x, tra.xu - resf * eps_ddmc_offset() * dx, dx, eps)) {
      // sample y and z
      tra.y = tra.yl + tra.rng_gen.drand() * resf * dy;
      tra.z = tra.zl + tra.rng_gen.drand() * resf * dz;
      // sample direction
      sample_face_iso_dir(-tra.vv, tra.rng_gen, tra.vx, tra.vy, tra.vz);
      // break loop
      res_valid = true;
      break;
    } else if (fuzzy_equal(tra.y, tra.yl + resf * eps_ddmc_offset() * dy, dy, eps) &&
               tra.multi_d) {
      // sample x and z
      tra.x = tra.xl + tra.rng_gen.drand() * resf * dx;
      tra.z = tra.zl + tra.rng_gen.drand() * resf * dz;
      // sample direction
      sample_face_iso_dir(tra.vv, tra.rng_gen, tra.vy, tra.vz, tra.vx);
      res_valid = true;
    } else if (fuzzy_equal(tra.y, tra.yu - resf * eps_ddmc_offset() * dy, dy, eps) &&
               tra.multi_d) {
      // sample x and z
      tra.x = tra.xl + tra.rng_gen.drand() * resf * dx;
      tra.z = tra.zl + tra.rng_gen.drand() * resf * dz;
      // sample direction
      sample_face_iso_dir(-tra.vv, tra.rng_gen, tra.vy, tra.vz, tra.vx);
      // break loop
      res_valid = true;
      break;
    } else if (fuzzy_equal(tra.z, tra.zl + resf * eps_ddmc_offset() * dz, dz, eps) &&
               tra.three_d) {
      // sample x and y
      tra.x = tra.xl + tra.rng_gen.drand() * resf * dx;
      tra.y = tra.yl + tra.rng_gen.drand() * resf * dy;
      // sample direction
      sample_face_iso_dir(tra.vv, tra.rng_gen, tra.vz, tra.vx, tra.vy);
      // break loop
      res_valid = true;
      break;
    } else if (fuzzy_equal(tra.z, tra.zu - resf * eps_ddmc_offset() * dz, dz, eps) &&
               tra.three_d) {
      // sample x and y
      tra.x = tra.xl + tra.rng_gen.drand() * resf * dx;
      tra.y = tra.yl + tra.rng_gen.drand() * resf * dy;
      // sample direction
      sample_face_iso_dir(-tra.vv, tra.rng_gen, tra.vz, tra.vx, tra.vy);
      // break loop
      res_valid = true;
      break;
    }
  }
  PARTHENON_REQUIRE(res_valid, "Invalid transition from DDMC to IMC.");
}

} // namespace jaybenne

#endif // JAYBENNE_TRANSPORT_UTILS_HPP_
