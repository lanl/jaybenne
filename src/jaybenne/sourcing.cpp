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

// Jaybenne includes
#include "jaybenne.hpp"
#include "jaybenne_utils.hpp"
#include "jaybenne_variables.hpp"

namespace jaybenne {

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus SourcePhotons
//! \brief Create photons, either during initialization or during a timestep.
//! TODO(BRR) modify interface so we don't need t_start, dt for initialization
template <typename T, SourceType ST, FrequencyType FT>
TaskStatus SourcePhotons(T *md, const Real t_start, const Real dt) {
  namespace fj = field::jaybenne;
  namespace fjh = field::jaybenne::host;
  namespace ph = particle::photons;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jb_pkg = pm->packages.Get("jaybenne");
  const Real h = jb_pkg->template Param<Real>("planck_constant");
  auto &eos = jb_pkg->template Param<EOS>("eos_d");
  int n_nubins = JaybenneNull<int>();
  Real numin = JaybenneNull<Real>();
  Real numax = JaybenneNull<Real>();
  MeanOpacity mopacity;
  Opacity opacity;
  if constexpr (FT == FrequencyType::gray) {
    mopacity = jb_pkg->template Param<MeanOpacity>("mopacity_d");
  } else if constexpr (FT == FrequencyType::multigroup) {
    opacity = jb_pkg->template Param<Opacity>("opacity_d");
    n_nubins = jb_pkg->template Param<int>("n_nubins");
    numin = jb_pkg->template Param<Real>("numin");
    numax = jb_pkg->template Param<Real>("numax");
  }
  auto &do_emission = jb_pkg->template Param<bool>("do_emission");
  auto &source_strategy = jb_pkg->template Param<SourceStrategy>("source_strategy");
  PARTHENON_REQUIRE(source_strategy != SourceStrategy::energy,
                    "Energy source strategy not implemented!");
  // TODO(BRR) replace with jaybenne param to disable emission
  if (ST == SourceType::emission && do_emission == false) {
    return TaskStatus::complete;
  }

  // Extract params
  auto &rng_pool = jb_pkg->template Param<RngPool>("rng_pool");
  const int &num_particles = jb_pkg->template Param<int>("num_particles");
  const Real &vv = jb_pkg->template Param<Real>("speed_of_light");
  const Real &sb = jb_pkg->template Param<Real>("stefan_boltzmann");

  // Create pack
  static auto desc =
      MakePackDescriptor<fjh::density, fjh::sie, fj::fleck_factor, fj::source_ew_per_cell,
                         fj::source_num_per_cell, fj::emission_cdf, fj::energy_delta>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // Indexing and dimensionality
  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);
  const int &nblocks = vmesh.GetNBlocks();

  // MeshBlock dimensions/size
  const int nx1 = ib.e - ib.s + 1;
  const int nx2 = jb.e - jb.s + 1;
  const int nx3 = kb.e - kb.s + 1;
  const int num_cells = nx1 * nx2 * nx3;
  const Real npc = static_cast<Real>(num_particles) / num_cells /
                   (nblocks * md->GetMeshPointer()->nbtotal);

  ParArray1D<int> nparticles("# particles per block", nblocks);
  ParArray2D<int> prefix_sum("prefix sums per block", nblocks, num_cells);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "SourcePhotons1", DevExecSpace(), 0, 0, 0, nblocks - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b) {
        auto &coords = vmesh.GetCoordinates(b);
        const Real &dv = coords.CellVolume(0, 0, 0);
        int block_sum = 0.0;
        par_reduce_inner(
            member, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            [&](const int &k, const int &j, const int &i, int &ntot) {
              auto rng_gen = rng_pool.get_state();
              // Compute erad
              const Real &rho = vmesh(b, fjh::density(), k, j, i);
              const Real &sie = vmesh(b, fjh::sie(), k, j, i);
              const Real temp = eos.TemperatureFromDensityInternalEnergy(rho, sie);
              [[maybe_unused]] const Real &sbd = sb;
              [[maybe_unused]] const Real &vvd = vv;
              [[maybe_unused]] const Real &dtd = dt;
              [[maybe_unused]] auto opac = opacity;
              [[maybe_unused]] auto mopac = mopacity;
              [[maybe_unused]] const auto numind = numin;
              [[maybe_unused]] const auto numaxd = numax;
              [[maybe_unused]] const auto n_nubinsd = n_nubins;
              Real erad = JaybenneNull<Real>();
              if constexpr (ST == SourceType::thermal) {
                erad = (4.0 * sbd / vvd) * std::pow(temp, 4.0) * dv;
              } else if constexpr (ST == SourceType::emission) {
                Real emis = JaybenneNull<Real>();
                if constexpr (FT == FrequencyType::gray) {
                  emis = mopac.Emissivity(rho, temp);
                } else if constexpr (FT == FrequencyType::multigroup) {
                  // Construct emission CDF
                  const Real dlnu = (std::log(numaxd) - std::log(numind)) / n_nubinsd;
                  // this is the mid-point of the 1st group in log-space:
                  // nu=exp(log(numin) + 0.5*dlnu)
                  Real nu = numind * std::exp(0.5 * dlnu);
                  Real dnu = nu * dlnu;
                  vmesh(b, fj::emission_cdf(0), k, j, i) =
                      opac.EmissivityPerNu(rho, temp, numind * std::exp(0.5 * dlnu)) *
                      dnu;
                  for (int n = 1; n < n_nubinsd; n++) {
                    // this is the mid-point of group n in log-space:
                    // nu=exp(log(numin)+(n+0.5)*dlnu)
                    nu = numind * std::exp((n + 0.5) * dlnu);
                    dnu = dlnu * nu;
                    vmesh(b, fj::emission_cdf(n), k, j, i) =
                        opac.EmissivityPerNu(rho, temp, nu) * dnu +
                        vmesh(b, fj::emission_cdf(n - 1), k, j, i);
                  }
                  emis = 0.0;
                  for (int n = 0; n < n_nubinsd; n++) {
                    const Real dnu = dlnu * numind * std::exp((n + 0.5) * dlnu);
                    // Get total emissivity
                    emis += vmesh(b, fj::emission_cdf(n), k, j, i);
                    // Normalize emission CDF
                    vmesh(b, fj::emission_cdf(n), k, j, i) /=
                        vmesh(b, fj::emission_cdf(n_nubinsd - 1), k, j, i);
                  }
                }
                erad = vmesh(b, fj::fleck_factor(), k, j, i) * emis * dv * dtd;
              }
              // Set source_num_per_cell
              Real &snpc = vmesh(b, fj::source_num_per_cell(), k, j, i);
              snpc = std::floor(npc);
              snpc += ((npc - snpc) > rng_gen.drand());
              ntot += static_cast<int>(std::round(snpc));
              vmesh(b, fj::source_ew_per_cell(), k, j, i) = erad / snpc;
              rng_pool.free_state(rng_gen);
            },
            Kokkos::Sum<int>(block_sum));
        Kokkos::single(Kokkos::PerTeam(member), [&]() { nparticles(b) = block_sum; });
        member.team_barrier();

        Kokkos::parallel_scan(Kokkos::TeamThreadRange(member, num_cells),
                              [&](const int &idx, int &update, const bool &finale) {
                                int k = idx / (nx1 * nx2) + kb.s;
                                int j = (idx / nx1) % nx2 + jb.s;
                                int i = idx % nx1 + ib.s;
                                if (finale) prefix_sum(b, idx) = update;
                                update += static_cast<int>(std::round(
                                    vmesh(b, fj::source_num_per_cell(), k, j, i)));
                              });
      });
  Kokkos::fence();

  // NOTE(PDM): Consider making mesh-level equivalents to the following
  ParArray1D<NewParticlesContext> new_contexts("New contexts", nblocks);
  auto new_contexts_h = new_contexts.GetHostMirror();
  auto nparticles_h = nparticles.GetHostMirrorAndCopy();
  for (int b = 0; b <= nblocks - 1; ++b) {
    auto particles = md->GetSwarmData(b)->Get(photons_swarm_name);
    new_contexts_h(b) = particles->AddEmptyParticles(nparticles_h(b));
  }
  new_contexts.DeepCopy(new_contexts_h);
  Kokkos::fence();

  static auto pdesc_r =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z,
                              ph::time, ph::v, ph::energy, ph::weight>(
          photons_swarm_name);
  static auto pdesc_i = MakeSwarmPackDescriptor<ph::ijk>(photons_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SourcePhotons2", parthenon::DevExecSpace(), 0, nblocks - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &coords = vmesh.GetCoordinates(b);
        const Real &xi = coords.template Xc<X1DIR>(i);
        const Real &yi = coords.template Xc<X2DIR>(j);
        const Real &zi = coords.template Xc<X3DIR>(k);
        const Real &dx_i = coords.template Dxc<X1DIR>(k, j, i);
        const Real &dx_j = coords.template Dxc<X2DIR>(k, j, i);
        const Real &dx_k = coords.template Dxc<X3DIR>(k, j, i);
        const Real x_min = coords.template Xc<parthenon::X1DIR>(ib.s) - 0.5 * dx_i;
        const Real y_min = coords.template Xc<parthenon::X2DIR>(jb.s) - 0.5 * dx_j;
        const Real z_min = coords.template Xc<parthenon::X3DIR>(kb.s) - 0.5 * dx_k;
        const int cell_idx_1d = (k - kb.s) * (nx1 * nx2) + (j - jb.s) * nx1 + (i - ib.s);
        auto rng_gen = rng_pool.get_state();
        [[maybe_unused]] const Real &dtd = dt;
        [[maybe_unused]] const Real &t_startd = t_start;
        [[maybe_unused]] const Real hd = h;
        [[maybe_unused]] const Real numaxd = numax;
        [[maybe_unused]] const Real numind = numin;
        [[maybe_unused]] const int n_nubinsd = n_nubins;

        // Starting index and length of particles in this cell
        const int &pstart_idx = prefix_sum(b, cell_idx_1d);
        const int num_part_per_cell =
            static_cast<int>(std::round(vmesh(b, fj::source_num_per_cell(), k, j, i)));

        Real &dejbn = vmesh(b, fj::energy_delta(), k, j, i);
        dejbn = 0.0;
        for (int np = pstart_idx; np < pstart_idx + num_part_per_cell; np++) {
          const int &n = new_contexts(b).GetNewParticleIndex(np);
          ppack_i(b, ph::ijk(0), n) = i;
          ppack_i(b, ph::ijk(1), n) = j;
          ppack_i(b, ph::ijk(2), n) = k;

          // Set energy weight
          ppack_r(b, ph::weight(), n) = vmesh(b, fj::source_ew_per_cell(), k, j, i);

          // Sample position uniformly in space over cell
          // TODO(BRR) only valid for Cartesian
          ppack_r(b, swarm_position::x(), n) = xi + dx_i * (rng_gen.drand() - 0.5);
          ppack_r(b, swarm_position::y(), n) = yi + dx_j * (rng_gen.drand() - 0.5);
          ppack_r(b, swarm_position::z(), n) = zi + dx_k * (rng_gen.drand() - 0.5);

          // Sample direction uniformly in solid angle
          const Real theta = std::acos(2.0 * rng_gen.drand() - 1.0);
          const Real phi = 2.0 * M_PI * rng_gen.drand();
          const Real stheta = std::sin(theta);
          ppack_r(b, ph::v(0), n) = vv * stheta * std::cos(phi);
          ppack_r(b, ph::v(1), n) = vv * stheta * std::sin(phi);
          ppack_r(b, ph::v(2), n) = vv * std::cos(theta);

          // Sample energy from Planck distribution
          // TODO(BRR) Extend to general frequency-structured emissivity
          const Real &rho = vmesh(b, fjh::density(), k, j, i);
          const Real &sie = vmesh(b, fjh::sie(), k, j, i);
          const Real temp = eos.TemperatureFromDensityInternalEnergy(rho, sie);
          if constexpr (FT == FrequencyType::gray) {
            ppack_r(b, ph::energy(), n) = sample_Planck_energy(rng_gen, sb, temp);
          } else if constexpr (FT == FrequencyType::multigroup) {
            // Sample energy from CDF
            const Real rand = rng_gen.drand();
            int n;
            for (n = 0; n < n_nubinsd; n++) {
              if (vmesh(b, fj::emission_cdf(n), k, j, i) >= rand) {
                break;
              }
            }
            const Real dlnu = (std::log(numaxd) - std::log(numind)) / n_nubinsd;
            const Real nu = numind * std::exp((n + 0.5) * dlnu);
            ppack_r(b, ph::energy(), n) = hd * nu;
          }

          if constexpr (ST == SourceType::emission) {
            dejbn -= ppack_r(b, ph::weight(), n);
            // Sample uniformly over timestep
            ppack_r(b, ph::time(), n) = t_startd + rng_gen.drand() * dtd;
          } else {
            ppack_r(b, ph::time(), n) = 0.;
          }
        }

        rng_pool.free_state(rng_gen);
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef MeshBlockData<Real> BD;
typedef MeshData<Real> D;
typedef SourceType ST;
typedef FrequencyType FT;
template TaskStatus SourcePhotons<BD, ST::thermal, FT::gray>(BD *md, const Real t0,
                                                             const Real dt);
template TaskStatus SourcePhotons<BD, ST::emission, FT::gray>(BD *md, const Real t0,
                                                              const Real dt);
template TaskStatus SourcePhotons<D, ST::thermal, FT::gray>(D *md, const Real t0,
                                                            const Real dt);
template TaskStatus SourcePhotons<D, ST::emission, FT::gray>(D *md, const Real t0,
                                                             const Real dt);
template TaskStatus SourcePhotons<BD, ST::thermal, FT::multigroup>(BD *md, const Real t0,
                                                                   const Real dt);
template TaskStatus SourcePhotons<BD, ST::emission, FT::multigroup>(BD *md, const Real t0,
                                                                    const Real dt);
template TaskStatus SourcePhotons<D, ST::thermal, FT::multigroup>(D *md, const Real t0,
                                                                  const Real dt);
template TaskStatus SourcePhotons<D, ST::emission, FT::multigroup>(D *md, const Real t0,
                                                                   const Real dt);

} // namespace jaybenne
