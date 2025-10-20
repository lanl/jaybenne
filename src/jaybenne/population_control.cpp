//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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

// Jaybenne includes
#include "jaybenne.hpp"
#include "jaybenne_utils.hpp"

namespace jaybenne {

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ControlPopulation
//! \brief Ensure particle population is near target user number of particles.
//! This currently uses a very simple strategy of killing a particle based on a
//! probability calculated from the number of active in excess of the number of source
//! particles. The expectation value is then the number of source particles: N_rm ~ P_rm *
//! N_act = (1 - N_src / N_act) * N_act = N_act - N_src The remaining particles then each
//! have their energy weight renormalized.
TaskStatus ControlPopulation(MeshData<Real> *md, const int ncycle, const int ncycle_out) {
  namespace fj = field::jaybenne;
  namespace ph = particle::photons;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jb_pkg = pm->packages.Get("jaybenne");
  auto &rng_pool = jb_pkg->template Param<RngPool>("rng_pool");
  const auto &diagnostic_level = jb_pkg->template Param<int>("diagnostic_level");

  // Create SparsePack
  static auto desc =
      MakePackDescriptor<fj::active_ew_per_cell, fj::active_num_per_cell,
                         fj::old_active_ew_per_cell, fj::source_num_per_cell>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // Create SwarmPacks
  static auto pdesc_r = MakeSwarmPackDescriptor<ph::weight>(photons_swarm_name);
  static auto pdesc_i = MakeSwarmPackDescriptor<ph::ijk>(photons_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  // Indexing and dimensionality
  const int &nparticles_per_pack = ppack_r.GetMaxFlatIndex();
  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);
  const int &nblocks = vmesh.GetNBlocks();

  //--------------------------------------------------------------------------------------
  // reset active particle count and energy weight per cell to 0
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::zero-active-counts-1",
      parthenon::DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        vmesh(b, fj::active_ew_per_cell(), k, j, i) = 0.0;
        vmesh(b, fj::active_num_per_cell(), k, j, i) = 0.0;
      });

  //--------------------------------------------------------------------------------------
  // aggregate active particle count and energy per cell
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::active-ptcl-counts-1", DevExecSpace(), 0,
      nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          // logical location and weight of particle
          const int &ip = ppack_i(b, ph::ijk(0), n);
          const int &jp = ppack_i(b, ph::ijk(1), n);
          const int &kp = ppack_i(b, ph::ijk(2), n);
          const Real &ww = ppack_r(b, ph::weight(), n);

          // add ew and 1 to active particle number at cell
          Real &actew = vmesh(b, fj::active_ew_per_cell(), kp, jp, ip);
          Real &actnum = vmesh(b, fj::active_num_per_cell(), kp, jp, ip);
          Kokkos::atomic_add(&actew, ww);
          Kokkos::atomic_add(&actnum, 1.0);
        }
      });

  //--------------------------------------------------------------------------------------
  // print total number of active particles at a certain diagnostic level
  if (diagnostic_level >= 1 && ncycle % ncycle_out == 0) {
    Real num_tot_active_old = 0.0;
    global_sum_reduce(
        nblocks, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        "ControlPopulation::report-tot-active-1",
        [&](const int &b, const int &k, const int &j, const int &i, Real &tota) {
          tota += vmesh(b, fj::active_num_per_cell(), k, j, i);
        },
        num_tot_active_old);

    // print total on rank 0
    if (Globals::my_rank == 0) {
      std::cout << "Post-IMC # active particles: " << num_tot_active_old << std::endl;
    }
  }

  //--------------------------------------------------------------------------------------
  // sample probability to mark particle for removal
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::active-ptcl-remove", DevExecSpace(), 0,
      nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          auto rng_gen = rng_pool.get_state();

          // logical location and weight of particle
          const int &ip = ppack_i(b, ph::ijk(0), n);
          const int &jp = ppack_i(b, ph::ijk(1), n);
          const int &kp = ppack_i(b, ph::ijk(2), n);

          // sample if particle is to be removed
          const Real &actnum = vmesh(b, fj::active_num_per_cell(), kp, jp, ip);
          const Real &srcnum = vmesh(b, fj::source_num_per_cell(), kp, jp, ip);
          // TODO: fix this arbitrary hard-coded threshold
          if (actnum > 4 * srcnum) {
            const Real rand = rng_gen.drand();
            if (rand * actnum < actnum - 4 * srcnum) {
              swarm_d.MarkParticleForRemoval(n);
            }
          }
          rng_pool.free_state(rng_gen);
        }
      });

  //--------------------------------------------------------------------------------------
  // removed marked particles
  for (int b = 0; b <= nblocks - 1; ++b) {
    md->GetSwarmData(b)->Get(photons_swarm_name)->RemoveMarkedParticles();
  }

  //--------------------------------------------------------------------------------------
  // reset active particle count per cell to 0
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::zero-active-counts-2",
      parthenon::DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        vmesh(b, fj::active_num_per_cell(), k, j, i) = 0.0;
        vmesh(b, fj::old_active_ew_per_cell(), k, j, i) = 0.0;
      });

  //--------------------------------------------------------------------------------------
  // store old active energy weight totals per cell, after particle removal
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::store-old-active-ew", DevExecSpace(), 0,
      nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          // logical location and weight of particle
          const int &ip = ppack_i(b, ph::ijk(0), n);
          const int &jp = ppack_i(b, ph::ijk(1), n);
          const int &kp = ppack_i(b, ph::ijk(2), n);
          const Real &ww = ppack_r(b, ph::weight(), n);

          // add ew and 1 to active particle number at cell
          Real &actew = vmesh(b, fj::old_active_ew_per_cell(), kp, jp, ip);
          Real &actnum = vmesh(b, fj::active_num_per_cell(), kp, jp, ip);
          Kokkos::atomic_add(&actew, ww);
          Kokkos::atomic_add(&actnum, 1.0);
        }
      });

  //--------------------------------------------------------------------------------------
  // print total number of active particles at a certain diagnostic level
  if (diagnostic_level >= 1 && ncycle % ncycle_out == 0) {
    Real num_tot_active = 0.0;
    global_sum_reduce(
        nblocks, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        "ControlPopulation::report-tot-active-2",
        [&](const int &b, const int &k, const int &j, const int &i, Real &tota) {
          tota += vmesh(b, fj::active_num_per_cell(), k, j, i);
        },
        num_tot_active);

    // print total on rank 0
    if (Globals::my_rank == 0) {
      std::cout << "Post-PopCon # active particles: " << num_tot_active << std::endl;
    }
  }

  //--------------------------------------------------------------------------------------
  // renormalize surviving particle energy weights, to conserve energy
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::renorm-ptcl-ew", DevExecSpace(), 0,
      nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          // logical location and weight of particle
          const int &ip = ppack_i(b, ph::ijk(0), n);
          const int &jp = ppack_i(b, ph::ijk(1), n);
          const int &kp = ppack_i(b, ph::ijk(2), n);
          Real &ww = ppack_r(b, ph::weight(), n);

          const Real &oldactew = vmesh(b, fj::old_active_ew_per_cell(), kp, jp, ip);
          const Real &actew = vmesh(b, fj::active_ew_per_cell(), kp, jp, ip);
          PARTHENON_DEBUG_REQUIRE(oldactew > 0.0, "Particle in cell with 0 particles!");
          ww *= (actew / oldactew);
        }
      });

  return TaskStatus::complete;
}

} // namespace jaybenne
