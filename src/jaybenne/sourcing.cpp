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

// C++ includes
#include <limits>

// Parthenon includes
#include <utils/robust.hpp>

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
  PARTHENON_INSTRUMENT
  namespace fj = field::jaybenne;
  namespace fjh = field::jaybenne::host;
  namespace ph = particle::photons;
  using singularity::photons::OpacityAveraging;
  using singularity::photons::Planck;
  using singularity::photons::Rosseland;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jb_pkg = pm->packages.Get("jaybenne");
  const Real h = jb_pkg->template Param<Real>("planck_constant");
  auto &eos = jb_pkg->template Param<EOS>("eos_d");
  int n_nubins = JaybenneNull<int>();
  Real dlnu = JaybenneNull<Real>();
  std::vector<Real> nu_grid = JaybenneNull<std::vector<Real>>();
  ParArray1D<Real> nu_bins;
  MeanOpacity mopacity = jb_pkg->template Param<MeanOpacity>("mopacity_d");
  if constexpr (FT == FrequencyType::multigroup) {
    n_nubins = jb_pkg->template Param<int>("n_nubins");
    dlnu = jb_pkg->template Param<Real>("dlnu");
    nu_grid = jb_pkg->template Param<std::vector<Real>>("nu_grid");
    nu_bins = ParArray1D<Real>("nu_bins", n_nubins);
    auto nu_bins_h = nu_bins.GetHostMirror();
    for (int n = 0; n < n_nubins; ++n) {
      nu_bins_h(n) = nu_grid[n];
    }
    nu_bins.DeepCopy(nu_bins_h);
  }
  auto &do_emission = jb_pkg->template Param<bool>("do_emission");
  auto &source_strategy = jb_pkg->template Param<SourceStrategy>("source_strategy");
  const auto &use_planck = jb_pkg->template Param<bool>("use_planck");
  const auto &use_rosseland = jb_pkg->template Param<bool>("use_rosseland");
  // set opacity mode for grey opacity
  const OpacityAveraging gmode = (use_planck && !use_rosseland) ? Planck : Rosseland;
  PARTHENON_REQUIRE(source_strategy != SourceStrategy::energy,
                    "Energy source strategy not implemented!");
  // TODO(BRR) replace with jaybenne param to disable emission
  if (ST == SourceType::emission && do_emission == false) {
    return TaskStatus::complete;
  }

  // Extract params
  auto &rng_pool = jb_pkg->template Param<RngPool>("rng_pool");
  const int &num_particles = jb_pkg->template Param<int>("num_particles");
  const Real &dnpc_min = jb_pkg->template Param<Real>("dnpc_min");
  const Real &emit_temp_th = jb_pkg->template Param<Real>("emit_temp_threshold");
  const Real &vv = jb_pkg->template Param<Real>("speed_of_light");
  const Real &sb = jb_pkg->template Param<Real>("stefan_boltzmann");
  const Real &kbolt = jb_pkg->template Param<Real>("boltzmann");

  // Create pack
  static auto desc =
      MakePackDescriptor<fjh::density, fjh::sie, fj::fleck_factor,
                         fj::source_num_per_cell, fj::source_ew_per_cell,
                         fj::delta_num_per_cell, fj::active_num_per_cell,
                         fj::active_ew_per_cell, fj::emission_cdf, fj::energy_delta>(
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

  Real npc = std::floor(static_cast<Real>(num_particles) /
                        (num_cells * md->GetMeshPointer()->nbtotal));

  // adjust particle number per cell up if threshold temperature is used
  if (emit_temp_th > 0.0 && ST == SourceType::emission) {
    // count the number of cells above the temperature threshold
    Real ncell_abv_th = 0.0;
    global_sum_reduce(
        "SourcePhotons::count-emitting-cells", DevExecSpace(), nblocks, kb.s, kb.e, jb.s,
        jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                      Real &totth) {
          const Real &rho = vmesh(b, fjh::density(), k, j, i);
          const Real &sie = vmesh(b, fjh::sie(), k, j, i);
          const Real temp = eos.TemperatureFromDensityInternalEnergy(rho, sie);
          totth += (temp > emit_temp_th ? 1.0 : 0.0);
        },
        ncell_abv_th, true);

    PARTHENON_REQUIRE(ncell_abv_th > 0.0,
                      "emission source but all cells below threshold temperature!");

    // calculate scaling factor on number per cell
    const Real npratio =
        static_cast<Real>(num_cells * md->GetMeshPointer()->nbtotal) / ncell_abv_th;
    PARTHENON_DEBUG_REQUIRE(npratio >= 1.0,
                            "more emitting cells than total cells in problem!");

    // upgrade number of particles per (emitting) cell
    npc = std::floor(npc * npratio);
  }

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
              // Compute erad
              const Real &rho = vmesh(b, fjh::density(), k, j, i);
              const Real &sie = vmesh(b, fjh::sie(), k, j, i);
              const Real temp = eos.TemperatureFromDensityInternalEnergy(rho, sie);
              [[maybe_unused]] const auto gmoded = gmode;
              [[maybe_unused]] const auto &sbd = sb;
              [[maybe_unused]] const auto &kboltd = kbolt;
              [[maybe_unused]] const auto hd = h;
              [[maybe_unused]] const auto &vvd = vv;
              [[maybe_unused]] const auto &dtd = dt;
              [[maybe_unused]] auto mopac = mopacity;
              [[maybe_unused]] const auto n_nubinsd = n_nubins;
              [[maybe_unused]] const auto dlnud = dlnu;
              [[maybe_unused]] const auto &nu_binsd = nu_bins;
              Real erad = JaybenneNull<Real>();
              if constexpr (ST == SourceType::thermal) {
                erad = (4.0 * sbd / vvd) * std::pow(temp, 4.0) * dv;
                // leverage emission_cdf for initial Planck sampling
                if constexpr (FT == FrequencyType::multigroup) {
                  // calculate bin width (assuming log bin width)
                  Real ee = hd * nu_binsd(0);
                  Real dee = dlnud * ee;
                  vmesh(b, fj::emission_cdf(0), k, j, i) =
                      jaybenne::midpoint_Planck(kboltd * temp, ee, dee);
                  for (int n = 1; n < n_nubinsd; n++) {
                    ee = hd * nu_binsd(n);
                    dee = dlnud * ee;
                    vmesh(b, fj::emission_cdf(n), k, j, i) =
                        jaybenne::midpoint_Planck(kboltd * temp, ee, dee) +
                        vmesh(b, fj::emission_cdf(n - 1), k, j, i);
                  }
                  for (int n = 0; n < n_nubinsd; n++) {
                    // Normalize emission CDF
                    vmesh(b, fj::emission_cdf(n), k, j, i) /=
                        vmesh(b, fj::emission_cdf(n_nubinsd - 1), k, j, i);
                  }
                }
              } else if constexpr (ST == SourceType::emission) {
                Real emis = JaybenneNull<Real>();
                if constexpr (FT == FrequencyType::gray) {
                  const Real T4 = SQR(SQR(temp));
                  const Real abs = mopac.AbsorptionCoefficient(rho, temp, 0, gmoded);
                  emis = abs * 4.0 * sbd * T4;
                } else if constexpr (FT == FrequencyType::multigroup) {
                  // Construct emission CDF
                  // calculate bin width (assuming log bin width)
                  // NOTE: frequency is used instead of group index to permit unequality
                  // between transport and MeanOpacity frequency grids (maybe not useful)
                  Real abs = mopac.AbsorptionCoefficient(rho, temp, nu_binsd(0), gmoded);
                  Real ee = hd * nu_binsd(0);
                  Real dee = dlnud * ee;
                  Real B = jaybenne::midpoint_Planck(kboltd * temp, ee, dee);
                  vmesh(b, fj::emission_cdf(0), k, j, i) = abs * B;
                  for (int n = 1; n < n_nubinsd; n++) {
                    abs = mopac.AbsorptionCoefficient(rho, temp, nu_binsd(n), gmoded);
                    ee = hd * nu_binsd(n);
                    dee = dlnud * ee;
                    B = jaybenne::midpoint_Planck(kboltd * temp, ee, dee);
                    vmesh(b, fj::emission_cdf(n), k, j, i) =
                        abs * B + vmesh(b, fj::emission_cdf(n - 1), k, j, i);
                  }
                  // Get total emissivity (before normalizing the CDF)
                  emis = vmesh(b, fj::emission_cdf(n_nubinsd - 1), k, j, i);
                  for (int n = 0; n < n_nubinsd; n++) {
                    // Normalize emission CDF
                    vmesh(b, fj::emission_cdf(n), k, j, i) /=
                        vmesh(b, fj::emission_cdf(n_nubinsd - 1), k, j, i);
                  }
                }
                erad = vmesh(b, fj::fleck_factor(), k, j, i) * emis * dv * dtd;
              }

              // Sourcing
              Real &snpc = vmesh(b, fj::source_num_per_cell(), k, j, i);
              Real &sewpc = vmesh(b, fj::source_ew_per_cell(), k, j, i);
              Real &actnum = vmesh(b, fj::active_num_per_cell(), k, j, i);
              Real &dnum = vmesh(b, fj::delta_num_per_cell(), k, j, i);
              // NOTE(PDM): We hardcode snpc and sewpc below, but we could imagine
              // introducing supplementary functions/tasks that set these, or even
              // giving downstream codes the opportunity to set these themselves...
              if (temp > emit_temp_th || ST == SourceType::thermal) {
                snpc = npc;
                dnum = std::max(std::round((snpc > actnum) * (snpc - actnum)), dnpc_min);
                sewpc = erad / dnum;
                ntot += static_cast<int>(dnum);
              } else {
                snpc = 0.0;
                dnum = 0.0;
                sewpc = 0.0;
              }
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
                                update += static_cast<int>(
                                    vmesh(b, fj::delta_num_per_cell(), k, j, i));
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
                              ph::time, ph::v, ph::energy, ph::weight, ph::fraction>(
          photons_swarm_name);
  static auto pdesc_i = MakeSwarmPackDescriptor<ph::ijk>(photons_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  constexpr Real eps = 4.0e8 * parthenon::robust::EPS();
  const Real ome = 1.0 - eps;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SourcePhotons2", parthenon::DevExecSpace(), 0, nblocks - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto rng_gen = rng_pool.get_state();
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
        [[maybe_unused]] const auto &kboltd = kbolt;
        [[maybe_unused]] const auto &dtd = dt;
        [[maybe_unused]] const auto &t_startd = t_start;
        [[maybe_unused]] const auto hd = h;
        [[maybe_unused]] const auto n_nubinsd = n_nubins;
        [[maybe_unused]] const auto dlnud = dlnu;
        [[maybe_unused]] const auto &nu_binsd = nu_bins;

        // Starting index and length of particles in this cell
        const int &pstart_idx = prefix_sum(b, cell_idx_1d);
        const int new_part_per_cell =
            static_cast<int>(vmesh(b, fj::delta_num_per_cell(), k, j, i));

        Real &dejbn = vmesh(b, fj::energy_delta(), k, j, i);
        dejbn = 0.0;
        for (int np = pstart_idx; np < pstart_idx + new_part_per_cell; np++) {
          const int &n = new_contexts(b).GetNewParticleIndex(np);
          ppack_i(b, ph::ijk(0), n) = i;
          ppack_i(b, ph::ijk(1), n) = j;
          ppack_i(b, ph::ijk(2), n) = k;

          // Set energy weight and fraction
          ppack_r(b, ph::weight(), n) = vmesh(b, fj::source_ew_per_cell(), k, j, i);
          ppack_r(b, ph::fraction(), n) = 1.0;

          // Sample position uniformly in space over cell
          // TODO(BRR) only valid for Cartesian
          ppack_r(b, swarm_position::x(), n) = xi + dx_i * ome * (rng_gen.drand() - 0.5);
          ppack_r(b, swarm_position::y(), n) = yi + dx_j * ome * (rng_gen.drand() - 0.5);
          ppack_r(b, swarm_position::z(), n) = zi + dx_k * ome * (rng_gen.drand() - 0.5);

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
            ppack_r(b, ph::energy(), n) = sample_Planck_energy(rng_gen, kboltd, temp);
          } else if constexpr (FT == FrequencyType::multigroup) {
            // Sample energy (particle frequency) from CDF
            const Real rand = rng_gen.drand();
            int g;
            for (g = 0; g < n_nubinsd; ++g) {
              if (vmesh(b, fj::emission_cdf(g), k, j, i) >= rand) {
                break;
              }
            }
            ppack_r(b, ph::energy(), n) = hd * nu_binsd(g);
          }

          if constexpr (ST == SourceType::emission) {
            // Sample uniformly over timestep
            ppack_r(b, ph::time(), n) = t_startd + rng_gen.drand() * dtd;
            dejbn -= ppack_r(b, ph::weight(), n);
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
template TaskStatus
SourcePhotons<MeshBlockData<Real>, SourceType::thermal, FrequencyType::gray>(
    MeshBlockData<Real> *md, const Real t0, const Real dt);
template TaskStatus
SourcePhotons<MeshData<Real>, SourceType::emission, FrequencyType::gray>(
    MeshData<Real> *md, const Real t0, const Real dt);
template TaskStatus
SourcePhotons<MeshBlockData<Real>, SourceType::thermal, FrequencyType::multigroup>(
    MeshBlockData<Real> *md, const Real t0, const Real dt);
template TaskStatus
SourcePhotons<MeshData<Real>, SourceType::emission, FrequencyType::multigroup>(
    MeshData<Real> *md, const Real t0, const Real dt);

} // namespace jaybenne
