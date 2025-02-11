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

// Parthenon includes
#include <utils/robust.hpp>

// Jaybenne includes
#include "jaybenne.hpp"
#include "jaybenne_utils.hpp"
#include "scattering.hpp"
#include "transport_utils.hpp"

namespace jaybenne {

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus TransportPhotons
//! \brief
template <FrequencyType FT>
TaskStatus TransportPhotons(MeshData<Real> *md, const Real t_start, const Real dt) {
  namespace fj = field::jaybenne;
  namespace fjh = field::jaybenne::host;
  namespace sp = swarm_position;
  namespace ph = particle::photons;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jb_pkg = pm->packages.Get("jaybenne");
  auto &eos = jb_pkg->template Param<EOS>("eos_d");
  Opacity opacity;
  Scattering scattering;
  MeanOpacity mopacity;
  MeanScattering mscattering;
  int n_nubins = JaybenneNull<int>();
  Real numin = JaybenneNull<Real>();
  Real numax = JaybenneNull<Real>();
  if constexpr (FT == FrequencyType::gray) {
    mopacity = jb_pkg->template Param<MeanOpacity>("mopacity_d");
    mscattering = jb_pkg->template Param<MeanScattering>("mscattering_d");
  } else if constexpr (FT == FrequencyType::multigroup) {
    opacity = jb_pkg->template Param<Opacity>("opacity_d");
    scattering = jb_pkg->template Param<Scattering>("scattering_d");
    n_nubins = jb_pkg->template Param<int>("n_nubins");
    numin = jb_pkg->template Param<Real>("numin");
    numax = jb_pkg->template Param<Real>("numax");
  }
  auto &rng_pool = jb_pkg->template Param<RngPool>("rng_pool");
  const Real vv = jb_pkg->template Param<Real>("speed_of_light");

  // Create SparsePack
  static auto desc =
      MakePackDescriptor<fjh::density, fjh::sie, fj::fleck_factor, fj::energy_delta>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // Create SwarmPacks
  static auto pdesc_r = MakeSwarmPackDescriptor<sp::x, sp::y, sp::z, ph::v, ph::energy,
                                                ph::weight, ph::time>(photons_swarm_name);
  static auto pdesc_i = MakeSwarmPackDescriptor<ph::ijk>(photons_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

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
      DEFAULT_LOOP_PATTERN, "TransportPhotons", DevExecSpace(), 0, nparticles_per_pack,
      KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          auto rng_gen = rng_pool.get_state();
          auto &coords = vmesh.GetCoordinates(b);
          const Real &dx_i = coords.DxcFA(X1DIR, 0, 0, 0);
          const Real &dx_j = coords.DxcFA(X2DIR, 0, 0, 0);
          const Real &dx_k = coords.DxcFA(X3DIR, 0, 0, 0);
          const Real dx_push = std::min(dx_i, std::min(dx_j, dx_k));

          // Particle properties
          Real &t = ppack_r(b, ph::time(), n);
          Real &vx = ppack_r(b, ph::v(0), n);
          Real &vy = ppack_r(b, ph::v(1), n);
          Real &vz = ppack_r(b, ph::v(2), n);
          const Real &ww = ppack_r(b, ph::weight(), n);
          const Real &ee = ppack_r(b, ph::energy(), n);

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

            // Calculate cell bounds
            const Real xl = coords.template Xc<parthenon::X1DIR>(ip) - 0.5 * dx_i;
            const Real xu = coords.template Xc<parthenon::X1DIR>(ip) + 0.5 * dx_i;
            const Real yl = coords.template Xc<parthenon::X2DIR>(jp) - 0.5 * dx_j;
            const Real yu = coords.template Xc<parthenon::X2DIR>(jp) + 0.5 * dx_j;
            const Real zl = coords.template Xc<parthenon::X3DIR>(kp) - 0.5 * dx_k;
            const Real zu = coords.template Xc<parthenon::X3DIR>(kp) + 0.5 * dx_k;

            // Extract physical quantities
            const Real &rho = vmesh(b, fjh::density(), kp, jp, ip);
            const Real &sie = vmesh(b, fjh::sie(), kp, jp, ip);
            const Real temp = eos.TemperatureFromDensityInternalEnergy(rho, sie);
            const Real &ff = vmesh(b, fj::fleck_factor(), kp, jp, ip);
            Real ss = JaybenneNull<Real>();
            Real aa = JaybenneNull<Real>();
            if constexpr (FT == FrequencyType::gray) {
              // TODO: use TotalScatteringCoefficient(rho, temp), when available
              ss = mscattering.RosselandMeanTotalScatteringCoefficient(rho, temp);
              aa = mopacity.AbsorptionCoefficient(rho, temp);
            } else if constexpr (FT == FrequencyType::multigroup) {
              ss = scattering.TotalScatteringCoefficient(rho, temp, ee);
              aa = opacity.AbsorptionCoefficient(rho, temp, ee);
            }

            // reset collision indicators
            bool is_absorbed = false;
            bool is_scattered = false;

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
                                t, x, y, z, is_absorbed, is_scattered};
            // clang-format on
            ptcl_transport_step(tra);
            swarm_d.Xtoijk(x, y, z, ip, jp, kp);

            //  If particle has left this block, drop out of transport loop for comms
            bool on_current_mesh_block;
            swarm_d.GetNeighborBlockIndex(n, x, y, z, on_current_mesh_block);
            if (!on_current_mesh_block) {
              PARTHENON_DEBUG_REQUIRE(!(is_absorbed || is_scattered),
                                      "Absorption/scattering event off block!");
              break;
            }

            if (is_absorbed) {
              // process absorption
              Real &dejbn = vmesh(b, fj::energy_delta(), kp, jp, ip);
              Kokkos::atomic_add(&dejbn, ww);
              swarm_d.MarkParticleForRemoval(n);
              break;
            }

            if (is_scattered) {
              // process scattering
              // TODO(BRR): if eff scatter, redistribute frequency
              // TODO(BRR): template on scattering model
              scatter(rng_gen, vv, vx, vy, vz);
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
//! \fn  TaskStatus CheckCompletion
//! \brief Checks all particles on this mesh to see if they have reached the end
//!        of the timestep. If not, further iterations of transport are indicated.
TaskStatus CheckCompletion(MeshData<Real> *md, const Real t_end) {
  namespace ph = particle::photons;

  // Create SwarmPacks
  static auto pdesc_r = MakeSwarmPackDescriptor<ph::time>(photons_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  const int &nparticles_per_pack = ppack_r.GetMaxFlatIndex();

  // TODO(BRR) do this reduction in the transport loop instead?

  int num_unfinished = 0;
  parthenon::par_reduce(
      "CheckCompletion", 0, nparticles_per_pack,
      KOKKOS_LAMBDA(const int idx, int &num_unfinished) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          if (ppack_r(b, ph::time(), n) < t_end) {
            num_unfinished++;
          }
        }
      },
      Kokkos::Sum<int>(num_unfinished));

  if (num_unfinished > 0) {
    return TaskStatus::iterate;
  } else {
    return TaskStatus::complete;
  }
}

//----------------------------------------------------------------------------------------
//! template instantiations
template TaskStatus TransportPhotons<FrequencyType::gray>(MeshData<Real> *md,
                                                          const Real t_start,
                                                          const Real dt);
template TaskStatus TransportPhotons<FrequencyType::multigroup>(MeshData<Real> *md,
                                                                const Real t_start,
                                                                const Real dt);

} // namespace jaybenne
