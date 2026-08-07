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

// Parthenon includes
#include <utils/robust.hpp>

// Jaybenne includes
#include "ddmc_mg_utils.hpp"
#include "jaybenne.hpp"
#include "jaybenne_utils.hpp"
#include "scattering.hpp"
#include "transport_utils.hpp"

namespace jaybenne {

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus TransportPhotons_DDMC
//! \brief
template <FrequencyType FT>
TaskStatus TransportPhotons_DDMC(MeshData<Real> *md, const Real t_start, const Real dt) {
  PARTHENON_INSTRUMENT
  namespace fj = field::jaybenne;
  namespace fjh = field::jaybenne::host;
  namespace sp = swarm_position;
  namespace ph = particle::photons;
  using TE = parthenon::TopologicalElement;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jb_pkg = pm->packages.Get("jaybenne");
  auto &eos = jb_pkg->template Param<EOS>("eos_d");
  auto &rng_pool = jb_pkg->template Param<RngPool>("rng_pool");
  const Real vv = jb_pkg->template Param<Real>("speed_of_light");
  const Real ske = 0.5 * SQR(vv);
  const Real &tau_ddmc = jb_pkg->template Param<Real>("tau_ddmc");
  const Real cutoff = jb_pkg->template Param<Real>("cutoff");

  // data needed for multigroup frequency sampling
  const Real h = jb_pkg->template Param<Real>("planck_constant");
  const Real hinv = 1.0 / h;
  const Real sb = jb_pkg->template Param<Real>("boltzmann");
  int n_nubins = JaybenneNull<int>();
  Real dlnu = JaybenneNull<Real>();
  std::vector<Real> nu_grid = JaybenneNull<std::vector<Real>>();
  ParArray1D<Real> nu_bins;
  Opacity opacity;
  Scattering scattering;
  if constexpr (FT == FrequencyType::multigroup) {
    n_nubins = jb_pkg->template Param<int>("n_nubins");
    // initialize (assumed) log-spaced frequency bins
    dlnu = jb_pkg->template Param<Real>("dlnu");
    nu_grid = jb_pkg->template Param<std::vector<Real>>("nu_grid");
    nu_bins = ParArray1D<Real>("nu_bins", n_nubins);
    auto nu_bins_h = nu_bins.GetHostMirror();
    for (int n = 0; n < n_nubins; ++n) {
      nu_bins_h(n) = nu_grid[n];
    }
    nu_bins.DeepCopy(nu_bins_h);
    // set opacity objects
    opacity = jb_pkg->template Param<Opacity>("opacity_d");
    scattering = jb_pkg->template Param<Scattering>("scattering_d");
  }

  // Create SparsePack
  static auto desc =
      MakePackDescriptor<fjh::density, fjh::sie, fj::emission_cdf, fj::fleck_factor,
                         fj::ddmc_lo_face_prob, fj::ddmc_hi_face_prob, fj::energy_delta,
                         fjh::absorption_opacity, fjh::scattering_opacity>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // Create SwarmPacks
  static auto pdesc_r =
      MakeSwarmPackDescriptor<sp::x, sp::y, sp::z, ph::v, ph::energy, ph::weight,
                              ph::fraction, ph::time>(photons_swarm_name);
  static auto pdesc_i = MakeSwarmPackDescriptor<ph::ijk>(photons_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  // set tolerance for checking particle 0-velocity (i.e. if from DDMC block)
  constexpr Real eps = parthenon::robust::EPS();

  // Indexing and dimensionality
  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;
  const bool multi_d = (ndim >= 2);
  const bool three_d = (ndim == 3);
  const int &nblocks = vmesh.GetNBlocks();
  const int &nparticles_per_pack = ppack_r.GetMaxFlatIndex();

  // History-based Monte Carlo
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "TransportPhotons_DDMC", DevExecSpace(), 0,
      nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          auto rng_gen = rng_pool.get_state();

          // frequency data, needed for multigroup
          [[maybe_unused]] const auto hd = h;
          [[maybe_unused]] const auto hinvd = hinv;
          [[maybe_unused]] const auto sbd = sb;
          [[maybe_unused]] const auto n_nubinsd = n_nubins;
          [[maybe_unused]] const auto dlnud = dlnu;
          [[maybe_unused]] const auto nu_binsd = nu_bins;

          auto &coords = vmesh.GetCoordinates(b);
          const Real &dx_i = coords.template Dxc<parthenon::X1DIR>(0, 0, 0);
          const Real &dx_j = coords.template Dxc<parthenon::X2DIR>(0, 0, 0);
          const Real &dx_k = coords.template Dxc<parthenon::X3DIR>(0, 0, 0);
          const Real dx_push = std::min(dx_i, std::min(dx_j, dx_k));

          // Particle properties
          Real &t = ppack_r(b, ph::time(), n);
          Real &vx = ppack_r(b, ph::v(0), n);
          Real &vy = ppack_r(b, ph::v(1), n);
          Real &vz = ppack_r(b, ph::v(2), n);
          Real &ww = ppack_r(b, ph::weight(), n);
          Real &fraction = ppack_r(b, ph::fraction(), n);
          Real &ee = ppack_r(b, ph::energy(), n);

          // Position and logical location of particle
          Real &x = ppack_r(b, sp::x(), n);
          Real &y = ppack_r(b, sp::y(), n);
          Real &z = ppack_r(b, sp::z(), n);
          int &ip = ppack_i(b, ph::ijk(0), n);
          int &jp = ppack_i(b, ph::ijk(1), n);
          int &kp = ppack_i(b, ph::ijk(2), n);
          // NOTE(@pdmullen): Update required following comms
          swarm_d.Xtoijk(x, y, z, ip, jp, kp);

          while (t < t_start + dt) {
            // Check sanity of physical and logical locations
            PARTHENON_DEBUG_REQUIRE(x >= swarm_d.x_min_ && x <= swarm_d.x_max_,
                                    "Particle initially outside block X1 domain!");
            PARTHENON_DEBUG_REQUIRE(y >= swarm_d.y_min_ && y <= swarm_d.y_max_,
                                    "Particle initially outside block X2 domain!");
            PARTHENON_DEBUG_REQUIRE(z >= swarm_d.z_min_ && z <= swarm_d.z_max_,
                                    "Particle initially outside block x3 domain!");
            PARTHENON_DEBUG_REQUIRE(ip >= ib.s && ip <= ib.e,
                                    "Particle initially outside X1 logical bnds!");
            PARTHENON_DEBUG_REQUIRE(jp >= jb.s && jp <= jb.e,
                                    "Particle initially outside X2 logical bnds!");
            PARTHENON_DEBUG_REQUIRE(kp >= kb.s && kp <= kb.e,
                                    "Particle initially outside X3 logical bnds!");

            // calculate cell bounds
            const Real xl = coords.template Xc<parthenon::X1DIR>(ip) - 0.5 * dx_i;
            const Real xu = coords.template Xc<parthenon::X1DIR>(ip) + 0.5 * dx_i;
            const Real yl = coords.template Xc<parthenon::X2DIR>(jp) - 0.5 * dx_j;
            const Real yu = coords.template Xc<parthenon::X2DIR>(jp) + 0.5 * dx_j;
            const Real zl = coords.template Xc<parthenon::X3DIR>(kp) - 0.5 * dx_k;
            const Real zu = coords.template Xc<parthenon::X3DIR>(kp) + 0.5 * dx_k;

            // Extract physical quantities
            const Real &ff = vmesh(b, fj::fleck_factor(), kp, jp, ip);
            Real ss = JaybenneNull<Real>();
            Real aa = JaybenneNull<Real>();
            [[maybe_unused]] Real rho = JaybenneNull<Real>();
            [[maybe_unused]] Real temp = JaybenneNull<Real>();
            [[maybe_unused]] auto opac = opacity;
            [[maybe_unused]] auto scatter = scattering;
            [[maybe_unused]] auto eost = eos;
            if constexpr (FT == FrequencyType::gray) {
              ss = vmesh(b, fjh::scattering_opacity(), kp, jp, ip);
              aa = vmesh(b, fjh::absorption_opacity(), kp, jp, ip);
            } else if constexpr (FT == FrequencyType::multigroup) {
              rho = vmesh(b, fjh::density(), kp, jp, ip);
              const Real &sie = vmesh(b, fjh::sie(), kp, jp, ip);
              temp = eost.TemperatureFromDensityInternalEnergy(rho, sie);
              ss = scatter.TotalScatteringCoefficient(rho, temp, hinvd * ee);
              aa = opac.AbsorptionCoefficient(rho, temp, hinvd * ee);
            }

            // reset collision indicators
            bool is_absorbed = false;
            bool is_scattered = false;
            bool is_rejected = false;
            bool is_census = false;

            const bool is_ddmc_step = dx_push * (ss + aa) > tau_ddmc;
            Real e_abs = 0.0;

            if (is_ddmc_step) {

              // sample if particle is undergoing an elastic event
              // NOTE: this uses the fact that the number of random walks in a cell scales
              // like tau^2, and at each event the probability of an elastic scatter is ss
              // / (ss + aa).
              bool is_elastic = false;
              if constexpr (FT == FrequencyType::multigroup) {
                const Real tau_min = dx_push * (ss + aa);
                is_elastic =
                    rng_gen.drand() < std::pow(ss / (ss + aa), tau_min * tau_min);
              }

              // Update cell of particle
              swarm_d.Xtoijk(x, y, z, ip, jp, kp);

              // calculate cell bounds
              const Real xl = coords.template Xc<parthenon::X1DIR>(ip) - 0.5 * dx_i;
              const Real xu = coords.template Xc<parthenon::X1DIR>(ip) + 0.5 * dx_i;
              const Real yl = coords.template Xc<parthenon::X2DIR>(jp) - 0.5 * dx_j;
              const Real yu = coords.template Xc<parthenon::X2DIR>(jp) + 0.5 * dx_j;
              const Real zl = coords.template Xc<parthenon::X3DIR>(kp) - 0.5 * dx_k;
              const Real zu = coords.template Xc<parthenon::X3DIR>(kp) + 0.5 * dx_k;

              // get face probabilities, if no absorption, use per-group values
              const Real Px_l =
                  is_elastic ? 1.0 / (3.0 * ss * dx_i)
                             : vmesh(b, TE::F1, fj::ddmc_hi_face_prob(), kp, jp, ip);
              const Real Px_u =
                  is_elastic ? 1.0 / (3.0 * ss * dx_i)
                             : vmesh(b, TE::F1, fj::ddmc_lo_face_prob(), kp, jp, ip + 1);
              const Real Py_l =
                  multi_d ? (is_elastic
                                 ? 1.0 / (3.0 * ss * dx_j)
                                 : vmesh(b, TE::F2, fj::ddmc_hi_face_prob(), kp, jp, ip))
                          : 0.0;
              const Real Py_u =
                  multi_d
                      ? (is_elastic
                             ? 1.0 / (3.0 * ss * dx_j)
                             : vmesh(b, TE::F2, fj::ddmc_lo_face_prob(), kp, jp + 1, ip))
                      : 0.0;
              const Real Pz_l =
                  three_d ? (is_elastic
                                 ? 1.0 / (3.0 * ss * dx_k)
                                 : vmesh(b, TE::F3, fj::ddmc_hi_face_prob(), kp, jp, ip))
                          : 0.0;
              const Real Pz_u =
                  three_d
                      ? (is_elastic
                             ? 1.0 / (3.0 * ss * dx_k)
                             : vmesh(b, TE::F3, fj::ddmc_lo_face_prob(), kp + 1, jp, ip))
                      : 0.0;

              // store old cell indices (this is only needed for multigroup)
              const int ip_old = ip;
              const int jp_old = jp;
              const int kp_old = kp;
              int ip_next = ip;
              int jp_next = jp;
              int kp_next = kp;

              // NOTE(MGDDMC): grey ss will be needed for inelastic scattering
              Real aa_g = aa;
              Real gm_g = 0.0;
              if constexpr (FT == FrequencyType::multigroup) {
                if (is_elastic) {
                  aa_g = 0.0;
                } else {
                  // clang-format off
                  ddmc_mg_cell_args dmgc{n_nubinsd,
                                         dlnud,
                                         hd,
                                         sbd,
                                         tau_ddmc,
                                         dx_push,
                                         rho,
                                         temp};

                  const auto aagm = calc_ddmc_mg_probs(opac, scatter, dmgc, nu_binsd);
                  aa_g = aagm.first;
                  gm_g = aagm.second;
                }
              }

              bool is_leaked = false;

              // create DDMC step argument list
              // clang-format off
              ddmc_step_args dia{ // constants
                                  rng_gen,
                                  t_start, dt,
                                  ff, aa_g, ss, gm_g, vv,
                                  multi_d, three_d,
                                  xl, yl, zl, xu, yu, zu,
                                  Px_l, Py_l, Pz_l, Px_u, Py_u, Pz_u,
                                  // updated by push
                                  t, x, y, z, vx, vy, vz,
                                  ip_next, jp_next, kp_next, ww, fraction, e_abs,
                                  is_absorbed, is_scattered, is_census, is_leaked};
              // clang-format on

              // check for IMC-DDMC albedo rejection if particle arrived from IMC region
              if (SQR(vx) + SQR(vy) + SQR(vz) > ske) ptcl_ddmc_albedo(dia, is_rejected);

              if (!is_rejected) ptcl_ddmc_step(dia, cutoff);

              if constexpr (FT == FrequencyType::multigroup) {

                if (is_census && !is_elastic) {
                  // clang-format off
                  ddmc_mg_cell_args dmgc{n_nubinsd,
                                         dlnud,
                                         hd,
                                         sbd,
                                         tau_ddmc,
                                         dx_push,
                                         rho,
                                         temp};
                  // clang-format on
                  // the final argument tells it to stay in DDMC groups
                  ee = sample_ddmc2imc_outscatter(opac, scatter, dmgc, nu_bins, rng_gen,
                                                  true);
                }

                // NOTE: these ddmc_mg_leak_args do not use adjacent cell, so
                // the face CDF does not sum here to ddmc_(hi|lo)_face_prob.
                // This is hopefully a minor error.

                // particle must have leaked if nothing else
                if (is_leaked && !is_elastic) {

                  // only one index should be +/-1 of the current index
                  const int ip_u = (ip_old == ip_next + 1 ? ip_old : ip_next);
                  const int ip_l = (ip_old == ip_next - 1 ? ip_old : ip_next);
                  const int jp_u = (jp_old == jp_next + 1 ? jp_old : jp_next);
                  const int jp_l = (jp_old == jp_next - 1 ? jp_old : jp_next);
                  const int kp_u = (kp_old == kp_next + 1 ? kp_old : kp_next);
                  const int kp_l = (kp_old == kp_next - 1 ? kp_old : kp_next);
                  PARTHENON_DEBUG_REQUIRE(ip_u + jp_u + kp_u - ip_l - jp_l - kp_l == 1,
                                          "invalid index difference for DDMC leakage");

                  // select side of face
                  const bool use_lo_x = (ip_old == ip_next - 1);
                  const bool use_lo_y = (jp_old == jp_next - 1);
                  const bool use_lo_z = (kp_old == kp_next - 1);
                  // use_lo can only be false here if one of the old is the new index+1
                  const bool use_lo = (use_lo_x || use_lo_y || use_lo_z);

                  // TODO(MGDDMC): is this dx alone sufficient for nu-sampling at face?
                  const Real dx_f =
                      (ip_old != ip_next ? dx_i : (jp_old != jp_next ? dx_j : dx_k));

                  // get rho and temperature
                  const Real &rho_l = vmesh(b, fjh::density(), kp_l, jp_l, ip_l);
                  const Real &sie_l = vmesh(b, fjh::sie(), kp_l, jp_l, ip_l);
                  const Real temp_l =
                      eos.TemperatureFromDensityInternalEnergy(rho_l, sie_l);
                  const Real &rho_u = vmesh(b, fjh::density(), kp_u, jp_u, ip_u);
                  const Real &sie_u = vmesh(b, fjh::sie(), kp_u, jp_u, ip_u);
                  const Real temp_u =
                      eos.TemperatureFromDensityInternalEnergy(rho_u, sie_u);

                  // clang-format off
                  const ddmc_mg_leak_args dmg{n_nubinsd,
                                              dlnud,
                                              hd,
                                              sbd,
                                              tau_ddmc,
                                              dx_f,
                                              dx_f,
                                              dx_f,
                                              dx_f,
                                              rho_l,
                                              rho_u,
                                              temp_l,
                                              temp_u};
                  // clang-format off

                  // sample particle frequency (energy units)
                  ee = sample_leakage_group(opac, scatter, dmg, nu_binsd, use_lo, rng_gen);
                }
              }

            } else {

              // push particle
              // clang-format off
              tran_step_args tra{ // constants
                                  rng_gen,
                                  t_start, dt,
                                  ff, aa, ss,
                                  vv, vx, vy, vz,
                                  dx_push, multi_d, three_d,
                                  xl, yl, zl, xu, yu, zu,
                                  // updated by push
                                  t, x, y, z, ww, fraction, e_abs,
                                  is_scattered, is_census, is_absorbed};

              // if v==0, particle is from a DDMC cell in another block at <= refinement
              if (SQR(vx) + SQR(vy) + SQR(vz) < 2.0 * eps * ske) ptcl_ddmc_to_imc(tra);
              PARTHENON_DEBUG_REQUIRE(SQR(vx) + SQR(vy) + SQR(vz) > ske,
                                      "Invalid velocity: lower than lightspeed");

              // clang-format on
              ptcl_transport_step(tra, cutoff);
            }

            // don't do an atomic if particle is not absorbed and deposits no energy
            // e.g., analog particle that reaches census or analog DDMC particle
            // exiting DDMC region
            if (e_abs > 0.0) {
              // process continuous absorption
              Real &dejbn = vmesh(b, fj::energy_delta(), kp, jp, ip);
              Kokkos::atomic_add(&dejbn, e_abs);
            }

            // continuous absorption with low cutoff allows particles to get to zero
            // energy weights, kill them so they don't lead to division by zero in
            // population control
            if (!(ww > 0.0)) {
              swarm_d.MarkParticleForRemoval(n);
              break;
            }

            // Update cell of particle
            swarm_d.Xtoijk(x, y, z, ip, jp, kp);

            // If particle has left this block, drop out of transport loop for comms
            bool on_current_mesh_block;
            swarm_d.GetNeighborBlockIndex(n, x, y, z, on_current_mesh_block);
            if (!on_current_mesh_block) {
              PARTHENON_DEBUG_REQUIRE(!(is_absorbed || is_scattered),
                                      "Absorption/scattering event off block!");
              const bool vmask = !(is_ddmc_step && multi_d && !is_rejected);
              vx *= vmask;
              vy *= vmask;
              vz *= vmask;
              break;
            }

            if (is_absorbed) {
              // process analog absorption or DDMC absorption
              Real &dejbn = vmesh(b, fj::energy_delta(), kp, jp, ip);
              Kokkos::atomic_add(&dejbn, ww);
              swarm_d.MarkParticleForRemoval(n);
              break;
            }

            if (is_scattered) {

              // form particle scattering argument struct
              ptcl_scat_args psa{rng_gen, vv, vx, vy, vz, ee};

              // if multigroup eff scatter, redistribute frequency
              if constexpr (FT == FrequencyType::gray) {

                // just do direction-sampling (only called by IMC, by construction)
                sample_vol_iso_dir(psa);

              } else if constexpr (FT == FrequencyType::multigroup) {

                if (is_ddmc_step) {
                  // clang-format off
                  ddmc_mg_cell_args dmgc{n_nubinsd,
                                         dlnud,
                                         hd,
                                         sbd,
                                         tau_ddmc,
                                         dx_push,
                                         rho,
                                         temp};
                  // clang-format on
                  ee = sample_ddmc2imc_outscatter(opac, scatter, dmgc, nu_bins, rng_gen);

                  // resample particle direction
                  sample_vol_iso_dir(psa);

                } else {

                  // form cell scattering argument struct
                  // clang-format off
                  cell_scat_args csa{b, ip, jp, kp,
                                     ff, aa, ss,
                                     n_nubinsd, hd};
                  // clang-format on

                  // invoke frequency-dependent scattering kernel
                  scatter_kernel(vmesh, csa, nu_binsd, psa);
                }
              }
            }

            if (is_census) {
              // reset fraction of particles that make it census
              ppack_r(b, ph::fraction(), n) = 1.0;
            }
          }
          rng_pool.free_state(rng_gen);
        }
      });

  for (int b = 0; b <= nblocks - 1; ++b) {
    md->GetSwarmData(b)->Get(photons_swarm_name)->RemoveMarkedParticles();
  }

  return TaskStatus::complete;
}
//----------------------------------------------------------------------------------------
//! template instantiations
template TaskStatus TransportPhotons_DDMC<FrequencyType::gray>(MeshData<Real> *md,
                                                               const Real t_start,
                                                               const Real dt);
template TaskStatus TransportPhotons_DDMC<FrequencyType::multigroup>(MeshData<Real> *md,
                                                                     const Real t_start,
                                                                     const Real dt);

} // namespace jaybenne
