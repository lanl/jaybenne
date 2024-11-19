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

// C++ includes
#include <limits>

// Jaybenne includes
#include "jaybenne.hpp"
#include "jaybenne_utils.hpp"

namespace jaybenne {

using TQ = TaskQualifier;

// TODO(BRR) Move these methods to Parthenon
TaskStatus MeshResetCommunication(MeshData<Real> *md) {
  const int nblocks = md->NumBlocks();
  for (int n = 0; n < nblocks; n++) {
    auto &mbd = md->GetBlockData(n);
    auto &sc = mbd->GetSwarmData();
    sc->ResetCommunication();
  }

  return TaskStatus::complete;
}

TaskStatus MeshSend(MeshData<Real> *md) {
  const int nblocks = md->NumBlocks();
  for (int n = 0; n < nblocks; n++) {
    auto &mbd = md->GetBlockData(n);
    auto &sc = mbd->GetSwarmData();
    sc->Send(BoundaryCommSubset::all);
  }

  return TaskStatus::complete;
}

TaskStatus MeshReceive(MeshData<Real> *md) {
  TaskStatus status = TaskStatus::complete;
  const int nblocks = md->NumBlocks();
  for (int n = 0; n < nblocks; n++) {
    auto &mbd = md->GetBlockData(n);
    auto &sc = mbd->GetSwarmData();
    auto local_status = sc->Receive(BoundaryCommSubset::all);
    if (local_status == TaskStatus::incomplete) {
      status = TaskStatus::incomplete;
    }
  }

  return status;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection jaybenne::RadiationStep
//! \brief Construct the collection of tasks that contribute to a complete radiation cycle
//!        from t to t + dt, including setting up derived quantities, sourcing particles,
//!        transporting particles, and communicating particles.
TaskCollection RadiationStep(Mesh *pmesh, const Real t_start, const Real dt) {
  namespace fj = field::jaybenne;

  auto &jb_pkg = pmesh->packages.Get("jaybenne");
  const auto &max_transport_iterations = jb_pkg->Param<int>("max_transport_iterations");
  const bool &use_ddmc = jb_pkg->Param<bool>("use_ddmc");

  // MeshData subsets
  auto ddmc_field_names = std::vector<std::string>{fj::ddmc_face_prob::name()};
  auto &ddmc_reg =
      pmesh->mesh_data.AddShallow("ddmc_reg", pmesh->mesh_data.Get(), ddmc_field_names);

  TaskCollection tc;
  TaskID none(0);
  const int num_partitions = pmesh->DefaultNumPartitions();
  PARTHENON_REQUIRE(
      num_partitions == 1,
      "Iterative tasking may not support multiple partitions per rank as of 2024/5/14")
  auto &reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = reg[i];
    // Get base register for particles and DDMC fields register (if applicable)
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    auto &md_ddmc = pmesh->mesh_data.GetOrAdd("ddmc_reg", i);

    // prepare for iterative transport loop
    auto derived = tl.AddTask(none, UpdateDerivedTransportFields, base.get(), dt);
    auto source = tl.AddTask(
        derived, jaybenne::SourcePhotons<MeshData<Real>, jaybenne::SourceType::emission>,
        base.get(), t_start, dt);
    auto bcs = use_ddmc ? parthenon::AddBoundaryExchangeTasks(source, tl, md_ddmc,
                                                              pmesh->multilevel)
                        : source;

    // keep pushing particles until there are none left
    auto [itl, push] = tl.AddSublist(bcs, {1, max_transport_iterations});
    auto transport =
        use_ddmc ? itl.AddTask(none, TransportPhotons_DDMC, base.get(), t_start, dt)
                 : itl.AddTask(none, TransportPhotons, base.get(), t_start, dt);
    auto reset_comms = itl.AddTask(transport, MeshResetCommunication, base.get());
    auto send = itl.AddTask(reset_comms, MeshSend, base.get());
    auto receive = itl.AddTask(transport | send, MeshReceive, base.get());
    auto sample_ddmc_bface =
        use_ddmc ? itl.AddTask(receive, SampleDDMCBlockFace, base.get()) : receive;
    auto complete =
        itl.AddTask(TQ::once_per_region | TQ::global_sync | TQ::completion,
                    sample_ddmc_bface, CheckCompletion, base.get(), t_start + dt);

    // Update radiation fields
    auto eval_rad =
        tl.AddTask(push, jaybenne::EvaluateRadiationEnergy<MeshData<Real>>, base.get());

    // Update fluid fields
    auto update_fluid = tl.AddTask(eval_rad, jaybenne::UpdateFluid, base.get());
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Jaybenne::Initialize
//! \brief Initialize the Jaybenne physics package. This function defines and sets the
//! parameters associated with Jaybenne, and enrolls the data variables associated with
//! this physics package.
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Opacity &opacity,
                                            Scattering &scattering, EOS &eos,
                                            std::string block_name) {
  auto pkg = std::make_shared<StateDescriptor>("jaybenne");

  // Total number of particles
  int num_particles = pin->GetInteger(block_name, "num_particles");
  pkg->AddParam<>("num_particles", num_particles);
  Real dt = pin->GetOrAddReal(block_name, "dt", std::numeric_limits<Real>::max());
  pkg->AddParam<>("dt", dt);

  // Minimum occupancy of swarm (measure of pool fragmentation) below which
  // defragmentation is triggered.
  Real min_swarm_occupancy = pin->GetOrAddReal(block_name, "min_swarm_occupancy", 0.);
  PARTHENON_REQUIRE(min_swarm_occupancy >= 0 && min_swarm_occupancy < 1.0,
                    "Minimum allowable swarm occupancy must be >= 0 and less than 1");
  pkg->AddParam<>("min_swarm_occupancy", min_swarm_occupancy);

  // Frequency range
  Real numin = pin->GetOrAddReal(block_name, "numin", std::numeric_limits<Real>::min());
  pkg->AddParam<>("numin", numin);
  Real numax = pin->GetOrAddReal(block_name, "numax", std::numeric_limits<Real>::max());
  pkg->AddParam<>("numax", numax);

  // Physical constants
  const auto units = opacity.GetRuntimePhysicalConstants();
  pkg->AddParam<>("speed_of_light", units.c);
  pkg->AddParam<>("stefan_boltzmann", units.sb);

  // RNG
  bool unique_rank_seeds = pin->GetOrAddBoolean(block_name, "unique_rank_seeds", true);
  pkg->AddParam<>("unique_rank_seeds", unique_rank_seeds);
  int seed = pin->GetOrAddInteger(block_name, "seed", 123);
  pkg->AddParam<>("seed", unique_rank_seeds ? seed + Globals::my_rank : seed);
  RngPool rng_pool(seed);
  pkg->AddParam<>("rng_pool", rng_pool);

  // Transport numerics
  int max_transport_iterations =
      pin->GetOrAddInteger(block_name, "max_transport_iterations", 10000);
  pkg->AddParam<>("max_transport_iterations", max_transport_iterations);

  // DDMC flag (0 = no DDMC)
  bool use_ddmc = pin->GetOrAddBoolean(block_name, "use_ddmc", false);
  pkg->AddParam<>("use_ddmc", use_ddmc);
  // parse or use default DDMC threshold = 5
  Real tau_ddmc = pin->GetOrAddReal(block_name, "tau_ddmc", 5.0);
  pkg->AddParam<>("tau_ddmc", tau_ddmc);

  // Sourcing strategy
  SourceStrategy source_strategy;
  std::string strategy = pin->GetOrAddString(block_name, "source_strategy", "uniform");
  if (strategy == "uniform") {
    source_strategy = SourceStrategy::uniform;
  } else if (strategy == "energy") {
    source_strategy = SourceStrategy::energy;
  } else {
    PARTHENON_FAIL("Only uniform or energy source strategies supported!");
  }
  pkg->AddParam<>("source_strategy", source_strategy);

  // Whether to include emission physics
  const bool do_emission = pin->GetOrAddBoolean(block_name, "do_emission", true);
  pkg->AddParam<>("do_emission", do_emission);

  // Whether to feedback on fluid
  const bool do_feedback = pin->GetOrAddBoolean(block_name, "do_feedback", true);
  pkg->AddParam<>("do_feedback", do_feedback);

  // Equation of state model
  pkg->AddParam<>("eos_d", eos.GetOnDevice());

  // Opacity model
  pkg->AddParam<>("opacity_d", opacity.GetOnDevice());

  // Scattering model
  pkg->AddParam<>("scattering_d", scattering.GetOnDevice());

  // Swarm and swarm variables
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  pkg->AddSwarm(photons_swarm_name, swarm_metadata);
  Metadata mreal({Metadata::Real});
  pkg->AddSwarmValue(particle::photons::time::name(), photons_swarm_name, mreal);
  pkg->AddSwarmValue(particle::photons::weight::name(), photons_swarm_name, mreal);
  pkg->AddSwarmValue(particle::photons::energy::name(), photons_swarm_name, mreal);
  Metadata mrealv({Metadata::Real, Metadata::Vector}, std::vector<int>{3});
  pkg->AddSwarmValue(particle::photons::v::name(), photons_swarm_name, mrealv);
  Metadata mintv({Metadata::Integer, Metadata::Vector}, std::vector<int>{3});
  pkg->AddSwarmValue(particle::photons::ijk::name(), photons_swarm_name, mintv);

  // Radiation fields
  Metadata m({Metadata::Cell, Metadata::Independent});
  pkg->AddField(field::jaybenne::energy_tally::name(), m);
  pkg->AddField(field::jaybenne::fleck_factor::name(), m);

  // Sourcing fields
  Metadata m_onecopy({Metadata::Cell, Metadata::OneCopy});
  pkg->AddField(field::jaybenne::source_ew_per_cell::name(), m_onecopy);
  pkg->AddField(field::jaybenne::source_num_per_cell::name(), m_onecopy);
  pkg->AddField(field::jaybenne::energy_delta::name(), m_onecopy);

  // Face-based radiation fields
  Metadata mface({Metadata::Face, Metadata::Derived, Metadata::FillGhost});
  pkg->AddField(field::jaybenne::ddmc_face_prob::name(), mface);

  // Radiation timestep
  pkg->EstimateTimestepMesh = EstimateTimestepMesh;

  return pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Jaybenne::EstimateTimestepMesh
//! \brief Compute radiation timestep
Real EstimateTimestepMesh(MeshData<Real> *md) {
  // TODO(BRR) This should be provided by mcblock or other downstream codes... jaybenne
  // should have no timestep constraint.
  return md->GetParentPointer()->packages.Get("jaybenne")->Param<Real>("dt");
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus UpdateDerivedTransportFields
//! \brief  fields set: Fleck factor, DDMC face probabilities
//!         Fleck factor formula:
//!         betaf = 4aT^3 / (rho * cv)
//!         f = 1 / (1 + betaf * opacP * c * dt)
//!         NOTE: if J = opacP * c * aR * T^4,
//!               then f = 1 / (1 + 4 * J * dt / (rho * cv * T))
TaskStatus UpdateDerivedTransportFields(MeshData<Real> *md, const Real dt) {
  namespace fj = field::jaybenne;
  namespace fjh = field::jaybenne::host;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jbn = pm->packages.Get("jaybenne");
  auto &eos = jbn->Param<EOS>("eos_d");
  auto &opacity = jbn->Param<Opacity>("opacity_d");

  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);

  static auto desc =
      MakePackDescriptor<fjh::density, fjh::sie, fj::fleck_factor, fj::ddmc_face_prob>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdateDerivedTransportFields::Fleck-Factor",
      parthenon::DevExecSpace(), 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // TODO: get the Planck mean opacity directly here?
        const Real &rho = vmesh(b, fjh::density(), k, j, i);
        const Real &sie = vmesh(b, fjh::sie(), k, j, i);
        const Real temp = eos.TemperatureFromDensityInternalEnergy(rho, sie);
        const Real cv = eos.SpecificHeatFromDensityInternalEnergy(rho, sie);
        const Real emis = opacity.Emissivity(rho, temp);
        vmesh(b, fj::fleck_factor(), k, j, i) =
            1.0 / (1.0 + (4.0 * emis / (rho * cv * temp)) * dt);
      });

  // if DDMC active, calculate symmetric (geom. invariant) portion of face probs
  const bool use_ddmc = jbn->Param<bool>("use_ddmc");
  if (use_ddmc) {

    // TODO: what units do face-based prolongation-restriction operations assume?
    // (Do the probabilities need to be divided by face area, for instance?)

    // define extrapolation distance (Habetler & Matkowski 1975)
    constexpr Real lam_ext = 0.7104;

    // get DDMC cell optical thickness threshold
    const Real tau_ddmc = jbn->Param<Real>("tau_ddmc");

    // get scattering to calculate DDMC threshold
    auto &scattering = jbn->Param<Scattering>("scattering_d");

    // calculate DDMC face probabilities in X1 direction
    const int iu = ib.e + 1;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "UpdateDerivedTransportFields::X1-DDMC-Prob",
        parthenon::DevExecSpace(), 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        iu, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          // get coordinates of block
          auto &coords = vmesh.GetCoordinates(b);
          const Real &dx_i = coords.DxcFA(parthenon::X1DIR, 0, 0, 0);

          // get current, lower, upper neighbor block levels in x-direction
          const Real rlev = static_cast<Real>(vmesh.GetLevel(b, 0, 0, 0));
          const Real rlev_lx = static_cast<Real>(vmesh.GetLevel(b, 0, 0, -1));
          const Real rlev_ux = static_cast<Real>(vmesh.GetLevel(b, 0, 0, 1));

          // calculate neighbor dx values
          const Real dx_lx = i == ib.s ? std::pow(2.0, rlev - rlev_lx) * dx_i : dx_i;
          const Real dx_ux = i == iu ? std::pow(2.0, rlev - rlev_ux) * dx_i : dx_i;

          // TODO: interpolate temperatures to evaluate face opacities?
          // (If opacity gradients are not large, maybe this is not needed)

          // for directions orthogonal to face direction, bound index at cell end vals ...
          // calculate opacity, scattering from lower and upper cell
          const Real &rho_l = vmesh(b, fjh::density(), k, j, i - 1);
          const Real &sie_l = vmesh(b, fjh::sie(), k, j, i - 1);
          const Real temp_l = eos.TemperatureFromDensityInternalEnergy(rho_l, sie_l);
          const Real &rho_u = vmesh(b, fjh::density(), k, j, i);
          const Real &sie_u = vmesh(b, fjh::sie(), k, j, i);
          const Real temp_u = eos.TemperatureFromDensityInternalEnergy(rho_u, sie_u);
          // TODO: replace 3rd argument when this routine operates in multigroup
          const Real ss_l = scattering.TotalScatteringCoefficient(rho_l, temp_l, 1.0);
          const Real aa_l = opacity.AbsorptionCoefficient(rho_l, temp_l, 1.0);
          const Real ss_u = scattering.TotalScatteringCoefficient(rho_u, temp_u, 1.0);
          const Real aa_u = opacity.AbsorptionCoefficient(rho_u, temp_u, 1.0);

          // calculate optical thicknesses from lower and upper cell
          Real tau_l = dx_lx * (ss_l + aa_l);
          Real tau_u = dx_ux * (ss_u + aa_u);
          tau_l = tau_l > tau_ddmc ? tau_l : 2.0 * lam_ext;
          tau_u = tau_u > tau_ddmc ? tau_u : 2.0 * lam_ext;

          // set probability (face DDMC albedo)
          vmesh(b, TE::F1, fj::ddmc_face_prob(), k, j, i) = 2.0 / (3.0 * (tau_l + tau_u));
        });

    // set face probabilities in X2 direction
    const bool multi_d = (pm->ndim > 1);
    if (multi_d) {
      const int ju = jb.e + 1;
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "UpdateDerivedTransportFields::X2-DDMC-Prob",
          parthenon::DevExecSpace(), 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, ju, ib.s,
          ib.e, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
            // get coordinates of block
            auto &coords = vmesh.GetCoordinates(b);
            const Real &dx_j = coords.DxcFA(parthenon::X2DIR, 0, 0, 0);

            // get current, lower, upper neighbor block levels in x-direction
            const Real rlev = static_cast<Real>(vmesh.GetLevel(b, 0, 0, 0));
            const Real rlev_ly = static_cast<Real>(vmesh.GetLevel(b, 0, -1, 0));
            const Real rlev_uy = static_cast<Real>(vmesh.GetLevel(b, 0, 1, 0));

            // calculate neighbor dx values
            const Real dx_ly = j == jb.s ? std::pow(2.0, rlev - rlev_ly) * dx_j : dx_j;
            const Real dx_uy = j == ju ? std::pow(2.0, rlev - rlev_uy) * dx_j : dx_j;

            // TODO: interpolate temperatures to evaluate face opacities?
            // (If opacity gradients are not large, maybe this is not needed)

            // calculate opacity, scattering from lower and upper cell
            const Real &rho_l = vmesh(b, fjh::density(), k, j - 1, i);
            const Real &sie_l = vmesh(b, fjh::sie(), k, j - 1, i);
            const Real temp_l = eos.TemperatureFromDensityInternalEnergy(rho_l, sie_l);
            const Real &rho_u = vmesh(b, fjh::density(), k, j, i);
            const Real &sie_u = vmesh(b, fjh::sie(), k, j, i);
            const Real temp_u = eos.TemperatureFromDensityInternalEnergy(rho_u, sie_u);
            // TODO: replace 3rd argument when this routine operates in multigroup
            const Real ss_l = scattering.TotalScatteringCoefficient(rho_l, temp_l, 1.0);
            const Real aa_l = opacity.AbsorptionCoefficient(rho_l, temp_l, 1.0);
            const Real ss_u = scattering.TotalScatteringCoefficient(rho_u, temp_u, 1.0);
            const Real aa_u = opacity.AbsorptionCoefficient(rho_u, temp_u, 1.0);

            // calculate optical thicknesses from lower and upper cell
            Real tau_l = dx_ly * (ss_l + aa_l);
            Real tau_u = dx_uy * (ss_u + aa_u);
            tau_l = tau_l > tau_ddmc ? tau_l : 2.0 * lam_ext;
            tau_u = tau_u > tau_ddmc ? tau_u : 2.0 * lam_ext;

            // set probability (face DDMC albedo)
            vmesh(b, TE::F2, fj::ddmc_face_prob(), k, j, i) =
                2.0 / (3.0 * (tau_l + tau_u));
          });
    }

    // set face probabilities in X3 direction
    const bool three_d = (pm->ndim > 2);
    if (three_d) {
      const int ku = kb.e + 1;
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "UpdateDerivedTransportFields::X3-DDMC-Prob",
          parthenon::DevExecSpace(), 0, md->NumBlocks() - 1, kb.s, ku, jb.s, jb.e, ib.s,
          ib.e, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
            // get coordinates of block
            auto &coords = vmesh.GetCoordinates(b);
            const Real &dx_k = coords.DxcFA(parthenon::X3DIR, 0, 0, 0);

            // get current, lower, upper neighbor block levels in x-direction
            const Real rlev = static_cast<Real>(vmesh.GetLevel(b, 0, 0, 0));
            const Real rlev_lz = static_cast<Real>(vmesh.GetLevel(b, -1, 0, 0));
            const Real rlev_uz = static_cast<Real>(vmesh.GetLevel(b, 1, 0, 0));

            // calculate neighbor dx values
            const Real dx_lz = k == kb.s ? std::pow(2.0, rlev - rlev_lz) * dx_k : dx_k;
            const Real dx_uz = k == ku ? std::pow(2.0, rlev - rlev_uz) * dx_k : dx_k;

            // TODO: interpolate temperatures to evaluate face opacities?
            // (If opacity gradients are not large, maybe this is not needed)

            // calculate opacity, scattering from lower and upper cell
            const Real &rho_l = vmesh(b, fjh::density(), k - 1, j, i);
            const Real &sie_l = vmesh(b, fjh::sie(), k - 1, j, i);
            const Real temp_l = eos.TemperatureFromDensityInternalEnergy(rho_l, sie_l);
            const Real &rho_u = vmesh(b, fjh::density(), k, j, i);
            const Real &sie_u = vmesh(b, fjh::sie(), k, j, i);
            const Real temp_u = eos.TemperatureFromDensityInternalEnergy(rho_u, sie_u);
            // TODO: replace 3rd argument when this routine operates in multigroup
            const Real ss_l = scattering.TotalScatteringCoefficient(rho_l, temp_l, 1.0);
            const Real aa_l = opacity.AbsorptionCoefficient(rho_l, temp_l, 1.0);
            const Real ss_u = scattering.TotalScatteringCoefficient(rho_u, temp_u, 1.0);
            const Real aa_u = opacity.AbsorptionCoefficient(rho_u, temp_u, 1.0);

            // calculate optical thicknesses from lower and upper cell
            Real tau_l = dx_lz * (ss_l + aa_l);
            Real tau_u = dx_uz * (ss_u + aa_u);
            tau_l = tau_l > tau_ddmc ? tau_l : 2.0 * lam_ext;
            tau_u = tau_u > tau_ddmc ? tau_u : 2.0 * lam_ext;

            // set probability (face DDMC albedo)
            vmesh(b, TE::F3, fj::ddmc_face_prob(), k, j, i) =
                2.0 / (3.0 * (tau_l + tau_u));
          });
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus TaskStatus DefragParticles(MeshBlock *pmb)
//! \brief  NOTE(PDM): currently unused???
//! TODO(BRR) We should re-enable this but add a runtime parameter that sets the
//! fractional fragmentation of the memory pool above which we defragment.
TaskStatus DefragParticles(MeshBlock *pmb) {
  auto &jbn = pmb->packages.Get("jaybenne");
  auto &min_swarm_occupancy = jbn->Param<Real>("min_swarm_occupancy");
  auto &swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get(photons_swarm_name);
  if (swarm->GetNumActive() > 0) {
    if (swarm->GetPackingEfficiency() < min_swarm_occupancy) {
      swarm->Defrag();
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus EvaluateRadiationEnergy
//! \brief
template <typename T>
TaskStatus EvaluateRadiationEnergy(T *md) {
  namespace fj = field::jaybenne;
  namespace ph = particle::photons;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jb_pkg = pm->packages.Get("jaybenne");

  // Create SparsePack
  static auto desc = MakePackDescriptor<fj::energy_tally>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // Create SwarmPacks
  static auto pdesc_r = MakeSwarmPackDescriptor<ph::weight>(photons_swarm_name);
  static auto pdesc_i = MakeSwarmPackDescriptor<ph::ijk>(photons_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  // Indexing and dimensionality
  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);
  const int &nblocks = vmesh.GetNBlocks();
  const int &nparticles_per_pack = ppack_r.GetMaxFlatIndex();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ZeroEnergyTally", parthenon::DevExecSpace(), 0, nblocks - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        vmesh(b, fj::energy_tally(), k, j, i) = 0.0;
      });

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "FillEnergyTally", DevExecSpace(), 0, nparticles_per_pack,
      KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          auto &coords = vmesh.GetCoordinates(b);
          const Real &dv = coords.template Volume<TopologicalElement::CC>();
          const int &ip = ppack_i(b, ph::ijk(0), n);
          const int &jp = ppack_i(b, ph::ijk(1), n);
          const int &kp = ppack_i(b, ph::ijk(2), n);
          Kokkos::atomic_add(&vmesh(b, fj::energy_tally(), kp, jp, ip),
                             ppack_r(b, ph::weight(), n) / dv);
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void InitializeRadiation
//! \brief Initialize radiation based on material temperature and either thermal or
//!        zero initial radiation.
void InitializeRadiation(MeshBlockData<Real> *mbd, const bool is_thermal) {
  if (is_thermal) {
    jaybenne::SourcePhotons<MeshBlockData<Real>, jaybenne::SourceType::thermal>(mbd, 0.0,
                                                                                0.0);
  }

  // Call this so radiation field variables are properly initialized for outputs
  jaybenne::EvaluateRadiationEnergy<MeshBlockData<Real>>(mbd);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus UpdateFluid
//! \brief
TaskStatus UpdateFluid(MeshData<Real> *md) {
  namespace fj = field::jaybenne;
  namespace fjh = field::jaybenne::host;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jb_pkg = pm->packages.Get("jaybenne");
  if (!(jb_pkg->Param<bool>("do_feedback"))) return TaskStatus::complete;

  // Create SparsePack
  static auto desc =
      MakePackDescriptor<fj::energy_delta, fjh::update_energy>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // Indexing and dimensionality
  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);
  const int &nblocks = vmesh.GetNBlocks();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdateFluid", parthenon::DevExecSpace(), 0, nblocks - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &coords = vmesh.GetCoordinates(b);
        const Real &dv = coords.Volume<TopologicalElement::CC>();
        Real &ee = vmesh(b, fjh::update_energy(), k, j, i);
        const Real delta = vmesh(b, fj::energy_delta(), k, j, i) / dv;
        ee += delta;
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
template TaskStatus EvaluateRadiationEnergy<MeshBlockData<Real>>(MeshBlockData<Real> *md);
template TaskStatus EvaluateRadiationEnergy<MeshData<Real>>(MeshData<Real> *md);

} // namespace jaybenne
