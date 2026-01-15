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

  PARTHENON_REQUIRE(FT == FrequencyType::gray, "DDMC only works in gray!");

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jb_pkg = pm->packages.Get("jaybenne");
  auto &rng_pool = jb_pkg->template Param<RngPool>("rng_pool");
  const Real vv = jb_pkg->template Param<Real>("speed_of_light");
  const Real ske = 0.5 * SQR(vv);
  const Real &tau_ddmc = jb_pkg->template Param<Real>("tau_ddmc");
  const Real cutoff = jb_pkg->template Param<Real>("cutoff");

  // Create SparsePack
  static auto desc =
      MakePackDescriptor<fj::fleck_factor, fj::ddmc_face_prob, fj::energy_delta,
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
            const Real &ss = vmesh(b, fjh::scattering_opacity(), kp, jp, ip);
            const Real &aa = vmesh(b, fjh::absorption_opacity(), kp, jp, ip);

            // reset collision indicators
            bool is_absorbed = false;
            bool is_scattered = false;
            bool is_rejected = false;
            bool is_census = false;

            const bool is_ddmc_step = dx_push * (ss + aa) > tau_ddmc;
            Real e_abs = 0.0;

            if (is_ddmc_step) {
              // Update cell of particle
              swarm_d.Xtoijk(x, y, z, ip, jp, kp);

              // calculate cell bounds
              const Real xl = coords.template Xc<parthenon::X1DIR>(ip) - 0.5 * dx_i;
              const Real xu = coords.template Xc<parthenon::X1DIR>(ip) + 0.5 * dx_i;
              const Real yl = coords.template Xc<parthenon::X2DIR>(jp) - 0.5 * dx_j;
              const Real yu = coords.template Xc<parthenon::X2DIR>(jp) + 0.5 * dx_j;
              const Real zl = coords.template Xc<parthenon::X3DIR>(kp) - 0.5 * dx_k;
              const Real zu = coords.template Xc<parthenon::X3DIR>(kp) + 0.5 * dx_k;

              // get face probabilities
              const Real &Px_l = vmesh(b, TE::F1, fj::ddmc_face_prob(), kp, jp, ip);
              const Real &Px_u = vmesh(b, TE::F1, fj::ddmc_face_prob(), kp, jp, ip + 1);
              const Real &Py_l =
                  multi_d ? vmesh(b, TE::F2, fj::ddmc_face_prob(), kp, jp, ip) : 0.0;
              const Real &Py_u =
                  multi_d ? vmesh(b, TE::F2, fj::ddmc_face_prob(), kp, jp + 1, ip) : 0.0;
              const Real &Pz_l =
                  three_d ? vmesh(b, TE::F3, fj::ddmc_face_prob(), kp, jp, ip) : 0.0;
              const Real &Pz_u =
                  three_d ? vmesh(b, TE::F3, fj::ddmc_face_prob(), kp + 1, jp, ip) : 0.0;

              // create DDMC step argument list
              // clang-format off
              ddmc_step_args dia{ // constants
                                  rng_gen,
                                  t_start, dt,
                                  ff, aa, ss, vv,
                                  multi_d, three_d,
                                  xl, yl, zl, xu, yu, zu,
                                  Px_l, Py_l, Pz_l, Px_u, Py_u, Pz_u,
                                  // updated by push
                                  t, x, y, z, vx, vy, vz,
                                  ip, jp, kp, is_absorbed, is_scattered, is_census};
              // clang-format on

              // check for IMC-DDMC albedo rejection if particle arrived from IMC region
              if (SQR(vx) + SQR(vy) + SQR(vz) > ske) ptcl_ddmc_albedo(dia, is_rejected);

              if (!is_rejected) ptcl_ddmc_step(dia);

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
                                  t, x, y, z, ww, fraction, e_abs, is_scattered, is_census, is_absorbed};

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
              // std::cout<<"zero weight particle hit killed block"<<std::endl;
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
              // process scattering
              // TODO(BRR): if eff scatter, redistribute frequency
              // TODO(BRR): template on scattering model
              ScatterKernel(rng_gen, vv, vx, vy, vz);
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
