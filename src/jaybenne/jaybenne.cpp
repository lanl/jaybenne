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

// Jaybenne includes
#include "ddmc_mg_utils.hpp"
#include "jaybenne.hpp"
#include "jaybenne_utils.hpp"
#include <utils/robust.hpp>

namespace jaybenne {

using TQ = TaskQualifier;

// TODO(BRR) Move these methods to Parthenon
TaskStatus MeshResetCommunication(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  const int nblocks = md->NumBlocks();
  for (int n = 0; n < nblocks; n++) {
    auto &mbd = md->GetBlockData(n);
    auto &sc = mbd->GetSwarmData();
    sc->ResetCommunication();
  }

  return TaskStatus::complete;
}

TaskStatus MeshSend(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  const int nblocks = md->NumBlocks();
  for (int n = 0; n < nblocks; n++) {
    auto &mbd = md->GetBlockData(n);
    auto &sc = mbd->GetSwarmData();
    sc->Send(BoundaryCommSubset::all);
  }

  return TaskStatus::complete;
}

TaskStatus MeshReceive(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
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
TaskCollection RadiationStep(Mesh *pmesh, const SimTime &tm, const Real dt) {
  PARTHENON_INSTRUMENT
  namespace fj = field::jaybenne;

  // short-cuts
  const Real &t_start = tm.time;
  const int &ncycle = tm.ncycle;
  const int &ncycle_out = tm.ncycle_out;
  PARTHENON_REQUIRE(fuzzy_equal(tm.dt, dt, dt, parthenon::robust::EPS()),
                    "Integrator dt (RadiationStep arg) must equal SimTime dt");

  auto &jb_pkg = pmesh->packages.Get("jaybenne");
  const auto &max_transport_iterations =
      jb_pkg->template Param<int>("max_transport_iterations");
  const bool &use_ddmc = jb_pkg->template Param<bool>("use_ddmc");
  const auto &fd = jb_pkg->template Param<FrequencyType>("frequency_type");

  // MeshData subsets
  auto ddmc_field_names = std::vector<std::string>{fj::ddmc_lo_face_prob::name(),
                                                   fj::ddmc_hi_face_prob::name()};
  auto &ddmc_reg =
      pmesh->mesh_data.AddShallow("ddmc_reg", pmesh->mesh_data.Get(), ddmc_field_names);

  TaskCollection tc;
  TaskID none(0);

  auto &timing_region0 = tc.AddRegion(1);
  {
    auto &tl = timing_region0[0];
    tl.AddTask(none, []() {
      Kokkos::Profiling::pushRegion("Jaybenne::Timestep");
      return TaskStatus::complete;
    });
  }

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
    auto source = derived;
    if (fd == FrequencyType::gray) {
      source = tl.AddTask(
          derived,
          jaybenne::SourcePhotons<MeshData<Real>, jaybenne::SourceType::emission,
                                  FrequencyType::gray>,
          base.get(), t_start, dt);
    } else if (fd == FrequencyType::multigroup) {
      source = tl.AddTask(
          derived,
          jaybenne::SourcePhotons<MeshData<Real>, jaybenne::SourceType::emission,
                                  FrequencyType::multigroup>,
          base.get(), t_start, dt);
    }
    auto bcs = use_ddmc ? parthenon::AddBoundaryExchangeTasks(source, tl, md_ddmc,
                                                              pmesh->multilevel)
                        : source;

    // keep pushing particles until there are none left
    auto [itl, push] = tl.AddSublist(bcs, {1, max_transport_iterations});
    auto time_start = itl.AddTask(none, []() {
      Kokkos::Profiling::pushRegion("Jaybenne::TransportLoop");
      return TaskStatus::complete;
    });
    auto transport = time_start;
    if (fd == FrequencyType::gray) {
      transport =
          use_ddmc ? itl.AddTask(time_start, TransportPhotons_DDMC<FrequencyType::gray>,
                                 base.get(), t_start, dt)
                   : itl.AddTask(time_start, TransportPhotons<FrequencyType::gray>,
                                 base.get(), t_start, dt);
    } else if (fd == FrequencyType::multigroup) {
      transport =
          use_ddmc
              ? itl.AddTask(time_start, TransportPhotons_DDMC<FrequencyType::multigroup>,
                            base.get(), t_start, dt)
              : itl.AddTask(time_start, TransportPhotons<FrequencyType::multigroup>,
                            base.get(), t_start, dt);
    }
    auto reset_comms = itl.AddTask(transport, MeshResetCommunication, base.get());
    auto send = itl.AddTask(reset_comms, MeshSend, base.get());
    auto receive = itl.AddTask(transport | send, MeshReceive, base.get());
    auto sample_ddmc_bface =
        use_ddmc ? itl.AddTask(receive, SampleDDMCBlockFace, base.get()) : receive;
    auto time_stop = itl.AddTask(sample_ddmc_bface, []() {
      Kokkos::Profiling::popRegion(/*Jaybenne::TransportLoop*/);
      return TaskStatus::complete;
    });
    auto complete = itl.AddTask(TQ::once_per_region | TQ::global_sync | TQ::completion,
                                time_stop, CheckCompletion, base.get(), t_start + dt);

    // Update radiation fields
    auto eval_rad =
        tl.AddTask(push, jaybenne::EvaluateRadiationEnergy<MeshData<Real>>, base.get());

    // Update fluid fields
    auto update_fluid = tl.AddTask(eval_rad, jaybenne::UpdateFluid, base.get());

    // Control particle population
    auto control_pop = tl.AddTask(update_fluid, jaybenne::ControlPopulation, base.get(),
                                  ncycle, ncycle_out);

    // TODO: Defrag particles? Verify parth swarm defrag mechanics before uncommenting
    // auto defrag_pop = tl.AddTask(control_pop, jaybenne::DefragParticles, base.get());
  }

  auto &timing_region1 = tc.AddRegion(1);
  {
    auto &tl = timing_region1[0];
    tl.AddTask(none, []() {
      Kokkos::Profiling::popRegion(/* Jaybenne::Timestep */);
      return TaskStatus::complete;
    });
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Jaybenne::Initialize_impl
//! \brief Initialize the Jaybenne physics package. This function defines and sets the
//! parameters associated with Jaybenne, and enrolls the data variables associated with
//! this physics package, for everything not related to frequency discretization.
std::shared_ptr<StateDescriptor>
Initialize_impl(ParameterInput *pin, EOS &eos,
                singularity::RuntimePhysicalConstants units, std::string block_name) {
  auto pkg = std::make_shared<StateDescriptor>("jaybenne");

  // Diagnostics verbosity
  int diagnostic_level = pin->GetOrAddInteger(block_name, "diagnostic_level", 0);
  pkg->AddParam<>("diagnostic_level", diagnostic_level);
  PARTHENON_REQUIRE(diagnostic_level >= 0 && diagnostic_level <= 1,
                    "diagnostic_level is currently restricted to 0 or 1");

  // Total number of particles
  int num_particles = pin->GetInteger(block_name, "num_particles");
  pkg->AddParam<>("num_particles", num_particles);
  Real dnpc_min = pin->GetOrAddReal(block_name, "dnpc_min", 20.0);
  PARTHENON_REQUIRE(dnpc_min >= 1.0, "dnpc_min must be at least 1");
  pkg->AddParam<>("dnpc_min", dnpc_min);
  Real dt = pin->GetOrAddReal(block_name, "dt", std::numeric_limits<Real>::max());
  pkg->AddParam<>("dt", dt);

  // Minimum occupancy of swarm (measure of pool fragmentation) below which
  // defragmentation is triggered.
  Real min_swarm_occupancy = pin->GetOrAddReal(block_name, "min_swarm_occupancy", 0.);
  PARTHENON_REQUIRE(min_swarm_occupancy >= 0 && min_swarm_occupancy < 1.0,
                    "Minimum allowable swarm occupancy must be >= 0 and less than 1");
  pkg->AddParam<>("min_swarm_occupancy", min_swarm_occupancy);

  // Physical constants
  // const auto units = opacity.GetRuntimePhysicalConstants();
  pkg->AddParam<>("speed_of_light", units.c);
  pkg->AddParam<>("stefan_boltzmann", units.sb);
  pkg->AddParam<>("planck_constant", units.h);
  pkg->AddParam<>("boltzmann", units.kb);

  // RNG
  bool unique_rank_seeds = pin->GetOrAddBoolean(block_name, "unique_rank_seeds", true);
  pkg->AddParam<>("unique_rank_seeds", unique_rank_seeds);
  int seed = pin->GetOrAddInteger(block_name, "seed", 123);
  if (unique_rank_seeds) {
    seed += Globals::my_rank;
  }
  pkg->AddParam<>("seed", seed);
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

  // parse or use defaul fractional weight cutoff (swtich to analog) of 1.0e-6
  Real cutoff = pin->GetOrAddReal(block_name, "cutoff", 1.0e-6);
  pkg->AddParam<>("cutoff", cutoff);

  // Select opacity average type to use (Planck/Rosseland)
  // if both true, then for grey runs an experimental Fleck(-Jiang) factor is used
  bool use_planck = pin->GetOrAddBoolean(block_name, "use_planck", false);
  pkg->AddParam<>("use_planck", use_planck);
  bool use_rosseland = pin->GetOrAddBoolean(block_name, "use_rosseland", true);
  pkg->AddParam<>("use_rosseland", use_rosseland);

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
  Real emit_temp_threshold = pin->GetOrAddReal(block_name, "emit_temp_threshold", 0.0);
  pkg->AddParam<>("emit_temp_threshold", emit_temp_threshold);

  // Whether to include emission physics
  const bool do_emission = pin->GetOrAddBoolean(block_name, "do_emission", true);
  pkg->AddParam<>("do_emission", do_emission);

  // Whether to feedback on fluid
  const bool do_feedback = pin->GetOrAddBoolean(block_name, "do_feedback", true);
  pkg->AddParam<>("do_feedback", do_feedback);

  // Equation of state model
  pkg->AddParam<>("eos_d", eos.GetOnDevice());

  // Swarm and swarm variables
  Metadata swarm_metadata({Metadata::Provides, Metadata::None, Metadata::Restart});
  pkg->AddSwarm(photons_swarm_name, swarm_metadata);
  Metadata mreal({Metadata::Real});
  pkg->AddSwarmValue(particle::photons::time::name(), photons_swarm_name, mreal);
  pkg->AddSwarmValue(particle::photons::weight::name(), photons_swarm_name, mreal);
  pkg->AddSwarmValue(particle::photons::fraction::name(), photons_swarm_name, mreal);
  pkg->AddSwarmValue(particle::photons::energy::name(), photons_swarm_name, mreal);
  Metadata mrealv({Metadata::Real, Metadata::Vector}, std::vector<int>{3});
  pkg->AddSwarmValue(particle::photons::v::name(), photons_swarm_name, mrealv);
  Metadata mintv({Metadata::Integer, Metadata::Vector}, std::vector<int>{3});
  pkg->AddSwarmValue(particle::photons::ijk::name(), photons_swarm_name, mintv);

  // Radiation fields
  Metadata m({Metadata::Cell, Metadata::Independent});
  pkg->AddField(field::jaybenne::energy_tally::name(), m);
  pkg->AddField(field::jaybenne::fleck_factor::name(), m);

  // Sourcing and tallying fields (recalculated each time step)
  Metadata m_onecopy({Metadata::Cell, Metadata::OneCopy});
  pkg->AddField(field::jaybenne::source_ew_per_cell::name(), m_onecopy);
  pkg->AddField(field::jaybenne::delta_num_per_cell::name(), m_onecopy);
  pkg->AddField(field::jaybenne::energy_delta::name(), m_onecopy);

  // Sourcing fields needed on restart
  Metadata m_onecopy_rst({Metadata::Cell, Metadata::OneCopy, Metadata::Restart});
  pkg->AddField(field::jaybenne::source_num_per_cell::name(), m_onecopy_rst);

  // Population control fields
  pkg->AddField(field::jaybenne::active_ew_per_cell::name(), m_onecopy);
  pkg->AddField(field::jaybenne::active_num_per_cell::name(), m_onecopy);
  pkg->AddField(field::jaybenne::old_active_ew_per_cell::name(), m_onecopy);

  // Face-based radiation fields
  Metadata mface({Metadata::Face, Metadata::Derived, Metadata::FillGhost});
  pkg->AddField(field::jaybenne::ddmc_lo_face_prob::name(), mface);
  pkg->AddField(field::jaybenne::ddmc_hi_face_prob::name(), mface);

  // Radiation timestep
  pkg->EstimateTimestepMesh = EstimateTimestepMesh;

  return pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Jaybenne::Initialize
//! \brief Initialize the Jaybenne physics package. This function defines and sets the
//! parameters associated with Jaybenne, and enrolls the data variables associated with
//! this physics package, specific to the gray frequency discretization.
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, MeanOpacity &mopacity,
                                            MeanScattering &mscattering, EOS &eos,
                                            std::string block_name) {
  const auto units = mopacity.GetRuntimePhysicalConstants();

  auto pkg = Initialize_impl(pin, eos, units, block_name);

  if (mopacity.ngroups() > 1 || mscattering.ngroups() > 1) {
    PARTHENON_REQUIRE(mopacity.ngroups() == mscattering.ngroups(),
                      "mopacity and mscattering have unequal group numbers");

    // Frequency discretization
    auto time = units.time;
    Real numin = pin->GetReal(block_name, "numin"); // in Hz
    Real numax = pin->GetReal(block_name, "numax"); // in Hz
    int n_nubins = pin->GetInteger(block_name, "n_nubins");
    pkg->AddParam<>("n_nubins", n_nubins);

    // assume units.time = [s/code time] = [code freq/Hz]
    numin *= time; // in code units
    numax *= time; // in code units

    // Construct and store frequency grid
    // NOTE: these are group interior points, not edges
    std::vector<Real> nu_grid(n_nubins, 0.0);
    // assume uniform log-spacing, grid is midpoints in log-space
    const Real dlnu = (std::log(numax) - std::log(numin)) / n_nubins;
    for (int n = 0; n < n_nubins; ++n) {
      nu_grid[n] = numin * std::exp((n + 0.5) * dlnu);
    }
    // store the grid in the parameter input
    pkg->AddParam<>("dlnu", dlnu);
    pkg->AddParam<>("nu_grid", nu_grid);

    // Emission CDF
    Metadata m_onecopy({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({n_nubins}));
    pkg->AddField(field::jaybenne::emission_cdf::name(), m_onecopy);

    pkg->AddParam<>("frequency_type", FrequencyType::multigroup);

  } else {
    // number of groups is 1 - assume gray
    pkg->AddParam<>("frequency_type", FrequencyType::gray);
  }

  // Opacity model
  pkg->AddParam<>("mopacity_d", mopacity.GetOnDevice());

  // Scattering model
  pkg->AddParam<>("mscattering_d", mscattering.GetOnDevice());

  return pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Jaybenne::EstimateTimestepMesh
//! \brief Compute radiation timestep
Real EstimateTimestepMesh(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  // TODO(BRR) This should be provided by mcblock or other downstream codes... jaybenne
  // should have no timestep constraint.
  return md->GetParentPointer()->packages.Get("jaybenne")->template Param<Real>("dt");
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus UpdateDerivedTransportFieldsImpl
//! \brief  fields set: Fleck factor, DDMC face probabilities
//!         Fleck factor formula:
//!         betaf = 4aT^3 / (rho * cv)
//!         f = 1 / (1 + betaf * opacP * c * dt)
//!         NOTE: if J = opacP * c * aR * T^4,
//!               then f = 1 / (1 + 4 * J * dt / (rho * cv * T))
template <FrequencyType FT>
TaskStatus UpdateDerivedTransportFieldsImpl(MeshData<Real> *md, const Real dt) {
  PARTHENON_INSTRUMENT
  namespace fj = field::jaybenne;
  namespace fjh = field::jaybenne::host;
  using singularity::photons::OpacityAveraging;
  using singularity::photons::Planck;
  using singularity::photons::Rosseland;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jbn = pm->packages.Get("jaybenne");
  auto &eos = jbn->template Param<EOS>("eos_d");
  MeanOpacity mopacity = jbn->template Param<MeanOpacity>("mopacity_d");
  ;
  MeanScattering mscattering = jbn->template Param<MeanScattering>("mscattering_d");
  int n_nubins = -1;
  Real dlnu = -1.0;
  Real h = -1.0;
  Real kbolt = -1.0;
  Real ac = -1.0; // radiation constant times light speed (set below)
  std::vector<Real> nu_grid = JaybenneNull<std::vector<Real>>();
  ParArray1D<Real> nu_bins;

  // get opacity average indicators
  const auto &use_planck = jbn->template Param<bool>("use_planck");
  const auto &use_rosseland = jbn->template Param<bool>("use_rosseland");
  // set opacity mode for emissivity used in Fleck factor
  const OpacityAveraging gmode = use_planck ? Planck : Rosseland;

  if constexpr (FT == FrequencyType::gray) {
    ac = 4.0 * (jbn->template Param<Real>("stefan_boltzmann"));
  } else if constexpr (FT == FrequencyType::multigroup) {
    PARTHENON_REQUIRE(!(use_planck && use_rosseland),
                      "Modified Fleck factor is not compatible with multigroup!");
    n_nubins = jbn->template Param<int>("n_nubins");
    h = jbn->template Param<Real>("planck_constant");
    kbolt = jbn->template Param<Real>("boltzmann");
    // initialize (assumed) log-spaced frequency grid
    dlnu = jbn->template Param<Real>("dlnu");
    nu_grid = jbn->template Param<std::vector<Real>>("nu_grid");
    nu_bins = ParArray1D<Real>("nu_bins", n_nubins);
    auto nu_bins_h = nu_bins.GetHostMirror();
    for (int n = 0; n < n_nubins; ++n) {
      nu_bins_h(n) = nu_grid[n];
    }
    nu_bins.DeepCopy(nu_bins_h);
  }

  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);

  static auto desc =
      MakePackDescriptor<fjh::density, fjh::sie, fj::fleck_factor, fj::ddmc_lo_face_prob,
                         fj::ddmc_hi_face_prob>(resolved_pkgs.get());
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
        Real emis = JaybenneNull<Real>();
        [[maybe_unused]] const auto acd = ac;
        [[maybe_unused]] const auto hd = h;
        [[maybe_unused]] const auto kboltd = kbolt;
        [[maybe_unused]] const auto gmoded = gmode;
        [[maybe_unused]] auto mopac = mopacity;
        [[maybe_unused]] const auto n_nubinsd = n_nubins;
        [[maybe_unused]] const auto &nu_binsd = nu_bins;
        [[maybe_unused]] const auto dlnud = dlnu;
        const Real acT4 = acd * SQR(SQR(temp));
        if constexpr (FT == FrequencyType::gray) {
          emis = mopac.AbsorptionCoefficient(rho, temp, 0, gmoded) * acT4;
        } else if constexpr (FT == FrequencyType::multigroup) {
          // NOTE: restriction to LTE emission here (pending generalization)
          emis = 0.0;
          Real plnk = 0.0;
          for (int n = 0; n < n_nubinsd; n++) {
            const Real abs =
                mopac.AbsorptionCoefficientFromNu(rho, temp, nu_binsd(n), gmoded);
            const Real ee = hd * nu_binsd(n);
            const Real dee = dlnud * ee;
            const Real B = jaybenne::midpoint_Planck(kboltd * temp, ee, dee);
            emis += abs * B;
            plnk += B;
          }
          PARTHENON_REQUIRE(plnk > 0.0, "Planck integral 0 in Fleck factor");
          // normalize
          emis /= plnk;
          emis *= acT4;
        }
        vmesh(b, fj::fleck_factor(), k, j, i) =
            1.0 / (1.0 + (4.0 * emis / (rho * cv * temp)) * dt);

        // check if alternate time-linearization is possible in this cell
        [[maybe_unused]] const bool use_pr = use_planck && use_rosseland;
        if constexpr (FT == FrequencyType::gray) {
          if (use_pr) {
            // calculate modified fleck factor using Planck and Rosseland
            const Real ross = mopac.AbsorptionCoefficient(rho, temp, 0, Rosseland);
            const Real plnk = mopac.AbsorptionCoefficient(rho, temp, 0, Planck);
            const Real &f = vmesh(b, fj::fleck_factor(), k, j, i);
            const Real fj = ross > 0.0 ? f * plnk / ross : f;

            // use factor (fj) only if <= 1 (ensure non-zero effective scattering)
            // if fj > 0, the original Fleck factor still has used Planck (gmode)
            vmesh(b, fj::fleck_factor(), k, j, i) = fj <= 1.0 ? fj : f;
          }
        }
      });

  // if DDMC active, calculate symmetric (geom. invariant) portion of face probs
  const bool use_ddmc = jbn->template Param<bool>("use_ddmc");
  if (use_ddmc) {

    // use Planck for all-Planck mode in leakage coefficients too
    const OpacityAveraging gmode2 = (use_planck && !use_rosseland) ? Planck : Rosseland;

    // define extrapolation distance (Habetler & Matkowski 1975)
    constexpr Real lam_ext = 0.7104;

    // get DDMC cell optical thickness threshold
    const Real tau_ddmc = jbn->template Param<Real>("tau_ddmc");

    // calculate DDMC face probabilities in X1 direction
    const int iu = ib.e + 1;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "UpdateDerivedTransportFields::X1-DDMC-Prob",
        parthenon::DevExecSpace(), 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        iu, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          // get coordinates of block
          auto &coords = vmesh.GetCoordinates(b);
          const Real &dx_i = coords.Dxc<parthenon::X1DIR>(0, 0, 0);
          const Real &dx_j = coords.Dxc<parthenon::X2DIR>(0, 0, 0);
          const Real &dx_k = coords.Dxc<parthenon::X3DIR>(0, 0, 0);

          // get current, lower, upper neighbor block levels in x-direction
          const Real rlev = static_cast<Real>(vmesh.GetLevel(b, 0, 0, 0));
          const Real rlev_lx = (vmesh.IsPhysicalBoundary(b, 0, 0, -1))
                                   ? rlev
                                   : static_cast<Real>(vmesh.GetLevel(b, 0, 0, -1));
          const Real rlev_ux = (vmesh.IsPhysicalBoundary(b, 0, 0, 1))
                                   ? rlev
                                   : static_cast<Real>(vmesh.GetLevel(b, 0, 0, 1));

          // calculate neighbor refinement scaling factors
          const Real scle_lx = i == ib.s ? std::pow(2.0, rlev - rlev_lx) : 1.0;
          const Real scle_ux = i == iu ? std::pow(2.0, rlev - rlev_ux) : 1.0;

          // calculate neighbor dx values
          const Real dx_lx = scle_lx * dx_i;
          const Real dx_ux = scle_ux * dx_i;

          // calculate neighbor min length for min optical thickness (for threshold check)
          const Real dx_push = std::min(dx_i, std::min(dx_j, dx_k));
          const Real dx_lmin = scle_lx * dx_push;
          const Real dx_umin = scle_ux * dx_push;

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
          Real ss_l = JaybenneNull<Real>();
          Real aa_l = JaybenneNull<Real>();
          Real ss_u = JaybenneNull<Real>();
          Real aa_u = JaybenneNull<Real>();
          [[maybe_unused]] const auto gmode2d = gmode2;
          [[maybe_unused]] auto mopac = mopacity;
          [[maybe_unused]] auto mscatter = mscattering;
          [[maybe_unused]] const auto dlnud = dlnu;
          [[maybe_unused]] const auto n_nubinsd = n_nubins;
          [[maybe_unused]] const auto &nu_binsd = nu_bins;
          [[maybe_unused]] const auto hd = h;
          [[maybe_unused]] const auto kboltd = kbolt;
          [[maybe_unused]] const auto tau_ddmcd = tau_ddmc;
          [[maybe_unused]] const auto lam_extd = lam_ext;
          if constexpr (FT == FrequencyType::gray) {
            ss_l = mscatter.ScatteringCoefficient(rho_l, temp_l, 0, gmode2d);
            aa_l = mopac.AbsorptionCoefficient(rho_l, temp_l, 0, gmode2d);
            ss_u = mscatter.ScatteringCoefficient(rho_u, temp_u, 0, gmode2d);
            aa_u = mopac.AbsorptionCoefficient(rho_u, temp_u, 0, gmode2d);

            // calculate optical thicknesses from lower and upper cell
            const Real tau_lmin = dx_lmin * (ss_l + aa_l);
            const Real tau_umin = dx_umin * (ss_u + aa_u);
            Real tau_l = dx_lx * (ss_l + aa_l);
            Real tau_u = dx_ux * (ss_u + aa_u);
            tau_l = tau_lmin > tau_ddmcd ? tau_l : 2.0 * lam_extd;
            tau_u = tau_umin > tau_ddmcd ? tau_u : 2.0 * lam_extd;

            // set probability (face DDMC albedo); for grey mode these are copies
            vmesh(b, TE::F1, fj::ddmc_lo_face_prob(), k, j, i) =
                2.0 / (3.0 * (tau_l + tau_u));
            vmesh(b, TE::F1, fj::ddmc_hi_face_prob(), k, j, i) =
                2.0 / (3.0 * (tau_l + tau_u));

          } else if constexpr (FT == FrequencyType::multigroup) {

            // create DDMC MG leakage data helper argument (defined in ddmc_mg_utils.hpp)
            // clang-format off
            const ddmc_mg_leak_args dmg{n_nubinsd,
                                        dlnud,
                                        hd,
                                        kboltd,
                                        tau_ddmcd,
                                        dx_lmin,
                                        dx_umin,
                                        dx_lx,
                                        dx_ux,
                                        rho_l,
                                        rho_u,
                                        temp_l,
                                        temp_u};
            // clang-format on

            // face side indicator
            constexpr bool use_lo = true;
            constexpr bool use_hi = false;

            // integrate lo-x leakage probability
            vmesh(b, TE::F1, fj::ddmc_lo_face_prob(), k, j, i) =
                calc_ddmc_mg_leakprob(mopac, mscatter, dmg, nu_binsd, use_lo);

            // integrate hi-x leakage probability
            vmesh(b, TE::F1, fj::ddmc_hi_face_prob(), k, j, i) =
                calc_ddmc_mg_leakprob(mopac, mscatter, dmg, nu_binsd, use_hi);
          }
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
            const Real &dx_i = coords.Dxc<parthenon::X1DIR>(0, 0, 0);
            const Real &dx_j = coords.Dxc<parthenon::X2DIR>(0, 0, 0);
            const Real &dx_k = coords.Dxc<parthenon::X3DIR>(0, 0, 0);

            // get current, lower, upper neighbor block levels in x-direction
            const Real rlev = static_cast<Real>(vmesh.GetLevel(b, 0, 0, 0));
            const Real rlev_ly = (vmesh.IsPhysicalBoundary(b, 0, -1, 0))
                                     ? rlev
                                     : static_cast<Real>(vmesh.GetLevel(b, 0, -1, 0));
            const Real rlev_uy = (vmesh.IsPhysicalBoundary(b, 0, 1, 0))
                                     ? rlev
                                     : static_cast<Real>(vmesh.GetLevel(b, 0, 1, 0));

            // calculate neighbor refinement scaling factors
            const Real scle_ly = j == jb.s ? std::pow(2.0, rlev - rlev_ly) : 1.0;
            const Real scle_uy = j == ju ? std::pow(2.0, rlev - rlev_uy) : 1.0;

            // calculate neighbor dx values
            const Real dx_ly = scle_ly * dx_j;
            const Real dx_uy = scle_uy * dx_j;

            // calculate neighbor min length for min optical thickness (for threshold
            // check)
            const Real dx_push = std::min(dx_i, std::min(dx_j, dx_k));
            const Real dx_lmin = scle_ly * dx_push;
            const Real dx_umin = scle_uy * dx_push;

            // TODO: interpolate temperatures to evaluate face opacities?
            // (If opacity gradients are not large, maybe this is not needed)

            // calculate opacity, scattering from lower and upper cell
            const Real &rho_l = vmesh(b, fjh::density(), k, j - 1, i);
            const Real &sie_l = vmesh(b, fjh::sie(), k, j - 1, i);
            const Real temp_l = eos.TemperatureFromDensityInternalEnergy(rho_l, sie_l);
            const Real &rho_u = vmesh(b, fjh::density(), k, j, i);
            const Real &sie_u = vmesh(b, fjh::sie(), k, j, i);
            const Real temp_u = eos.TemperatureFromDensityInternalEnergy(rho_u, sie_u);
            Real ss_l = JaybenneNull<Real>();
            Real aa_l = JaybenneNull<Real>();
            Real ss_u = JaybenneNull<Real>();
            Real aa_u = JaybenneNull<Real>();
            [[maybe_unused]] const auto gmode2d = gmode2;
            [[maybe_unused]] auto mopac = mopacity;
            [[maybe_unused]] auto mscatter = mscattering;
            [[maybe_unused]] const auto dlnud = dlnu;
            [[maybe_unused]] const auto n_nubinsd = n_nubins;
            [[maybe_unused]] const auto &nu_binsd = nu_bins;
            [[maybe_unused]] const auto hd = h;
            [[maybe_unused]] const auto kboltd = kbolt;
            [[maybe_unused]] const auto tau_ddmcd = tau_ddmc;
            [[maybe_unused]] const auto lam_extd = lam_ext;
            if constexpr (FT == FrequencyType::gray) {
              ss_l = mscatter.ScatteringCoefficient(rho_l, temp_l, 0, gmode2d);
              aa_l = mopac.AbsorptionCoefficient(rho_l, temp_l, 0, gmode2d);
              ss_u = mscatter.ScatteringCoefficient(rho_u, temp_u, 0, gmode2d);
              aa_u = mopac.AbsorptionCoefficient(rho_u, temp_u, 0, gmode2d);

              // calculate optical thicknesses from lower and upper cell
              const Real tau_lmin = dx_lmin * (ss_l + aa_l);
              const Real tau_umin = dx_umin * (ss_u + aa_u);
              Real tau_l = dx_ly * (ss_l + aa_l);
              Real tau_u = dx_uy * (ss_u + aa_u);
              tau_l = tau_lmin > tau_ddmcd ? tau_l : 2.0 * lam_extd;
              tau_u = tau_umin > tau_ddmcd ? tau_u : 2.0 * lam_extd;

              // set probability (face DDMC albedo); for grey mode these are copies
              vmesh(b, TE::F2, fj::ddmc_lo_face_prob(), k, j, i) =
                  2.0 / (3.0 * (tau_l + tau_u));
              vmesh(b, TE::F2, fj::ddmc_hi_face_prob(), k, j, i) =
                  2.0 / (3.0 * (tau_l + tau_u));

            } else if constexpr (FT == FrequencyType::multigroup) {

              // create DDMC MG leakage data helper argument (defined in
              // ddmc_mg_utils.hpp)
              // clang-format off
              const ddmc_mg_leak_args dmg{n_nubins,
                                          dlnud,
                                          hd,
                                          kboltd,
                                          tau_ddmcd,
                                          dx_lmin,
                                          dx_umin,
                                          dx_ly,
                                          dx_uy,
                                          rho_l,
                                          rho_u,
                                          temp_l,
                                          temp_u};
              // clang-format on

              // face side indicator
              constexpr bool use_lo = true;
              constexpr bool use_hi = false;

              // integrate lo-y leakage probability
              vmesh(b, TE::F2, fj::ddmc_lo_face_prob(), k, j, i) =
                  calc_ddmc_mg_leakprob(mopac, mscatter, dmg, nu_binsd, use_lo);

              // integrate hi-y leakage probability
              vmesh(b, TE::F2, fj::ddmc_hi_face_prob(), k, j, i) =
                  calc_ddmc_mg_leakprob(mopac, mscatter, dmg, nu_binsd, use_hi);
            }
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
            const Real &dx_i = coords.Dxc<parthenon::X1DIR>(0, 0, 0);
            const Real &dx_j = coords.Dxc<parthenon::X2DIR>(0, 0, 0);
            const Real &dx_k = coords.Dxc<parthenon::X3DIR>(0, 0, 0);

            // get current, lower, upper neighbor block levels in x-direction
            const Real rlev = static_cast<Real>(vmesh.GetLevel(b, 0, 0, 0));
            const Real rlev_lz = (vmesh.IsPhysicalBoundary(b, -1, 0, 0))
                                     ? rlev
                                     : static_cast<Real>(vmesh.GetLevel(b, -1, 0, 0));
            const Real rlev_uz = (vmesh.IsPhysicalBoundary(b, 1, 0, 0))
                                     ? rlev
                                     : static_cast<Real>(vmesh.GetLevel(b, 1, 0, 0));

            // calculate neighbor refinement scaling factors
            const Real scle_lz = k == kb.s ? std::pow(2.0, rlev - rlev_lz) : 1.0;
            const Real scle_uz = k == ku ? std::pow(2.0, rlev - rlev_uz) : 1.0;

            // calculate neighbor dx values
            const Real dx_lz = scle_lz * dx_k;
            const Real dx_uz = scle_uz * dx_k;

            // calculate neighbor min length for min optical thickness (for threshold
            // check)
            const Real dx_push = std::min(dx_i, std::min(dx_j, dx_k));
            const Real dx_lmin = scle_lz * dx_push;
            const Real dx_umin = scle_uz * dx_push;

            // TODO: interpolate temperatures to evaluate face opacities?
            // (If opacity gradients are not large, maybe this is not needed)

            // calculate opacity, scattering from lower and upper cell
            const Real &rho_l = vmesh(b, fjh::density(), k - 1, j, i);
            const Real &sie_l = vmesh(b, fjh::sie(), k - 1, j, i);
            const Real temp_l = eos.TemperatureFromDensityInternalEnergy(rho_l, sie_l);
            const Real &rho_u = vmesh(b, fjh::density(), k, j, i);
            const Real &sie_u = vmesh(b, fjh::sie(), k, j, i);
            const Real temp_u = eos.TemperatureFromDensityInternalEnergy(rho_u, sie_u);
            Real ss_l = JaybenneNull<Real>();
            Real aa_l = JaybenneNull<Real>();
            Real ss_u = JaybenneNull<Real>();
            Real aa_u = JaybenneNull<Real>();
            [[maybe_unused]] const auto gmode2d = gmode2;
            [[maybe_unused]] auto mopac = mopacity;
            [[maybe_unused]] auto mscatter = mscattering;
            [[maybe_unused]] const auto dlnud = dlnu;
            [[maybe_unused]] const auto n_nubinsd = n_nubins;
            [[maybe_unused]] const auto &nu_binsd = nu_bins;
            [[maybe_unused]] const auto hd = h;
            [[maybe_unused]] const auto kboltd = kbolt;
            [[maybe_unused]] const auto tau_ddmcd = tau_ddmc;
            [[maybe_unused]] const auto lam_extd = lam_ext;
            if constexpr (FT == FrequencyType::gray) {
              ss_l = mscatter.ScatteringCoefficient(rho_l, temp_l, 0, gmode2d);
              aa_l = mopac.AbsorptionCoefficient(rho_l, temp_l, 0, gmode2d);
              ss_u = mscatter.ScatteringCoefficient(rho_u, temp_u, 0, gmode2d);
              aa_u = mopac.AbsorptionCoefficient(rho_u, temp_u, 0, gmode2d);

              // calculate optical thicknesses from lower and upper cell
              const Real tau_lmin = dx_lmin * (ss_l + aa_l);
              const Real tau_umin = dx_umin * (ss_u + aa_u);
              Real tau_l = dx_lz * (ss_l + aa_l);
              Real tau_u = dx_uz * (ss_u + aa_u);
              tau_l = tau_lmin > tau_ddmcd ? tau_l : 2.0 * lam_extd;
              tau_u = tau_umin > tau_ddmcd ? tau_u : 2.0 * lam_extd;

              // set probability (face DDMC albedo); for grey mode these are copies
              vmesh(b, TE::F3, fj::ddmc_lo_face_prob(), k, j, i) =
                  2.0 / (3.0 * (tau_l + tau_u));
              vmesh(b, TE::F3, fj::ddmc_hi_face_prob(), k, j, i) =
                  2.0 / (3.0 * (tau_l + tau_u));

            } else if constexpr (FT == FrequencyType::multigroup) {

              // create DDMC MG leakage data helper argument (defined in
              // ddmc_mg_utils.hpp)
              // clang-format off
              const ddmc_mg_leak_args dmg{n_nubins,
                                          dlnud,
                                          hd,
                                          kboltd,
                                          tau_ddmcd,
                                          dx_lmin,
                                          dx_umin,
                                          dx_lz,
                                          dx_uz,
                                          rho_l,
                                          rho_u,
                                          temp_l,
                                          temp_u};
              // clang-format on

              // face side indicator
              constexpr bool use_lo = true;
              constexpr bool use_hi = false;

              // integrate lo-z leakage probability
              vmesh(b, TE::F3, fj::ddmc_lo_face_prob(), k, j, i) =
                  calc_ddmc_mg_leakprob(mopac, mscatter, dmg, nu_binsd, use_lo);

              // integrate hi-z leakage probability
              vmesh(b, TE::F3, fj::ddmc_hi_face_prob(), k, j, i) =
                  calc_ddmc_mg_leakprob(mopac, mscatter, dmg, nu_bins, use_hi);
            }
          });
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus UpdateDerivedTransportFieldsImpl
//! \brief  fields set: Fleck factor, DDMC face probabilities
//!         Fleck factor formula:
//!         betaf = 4aT^3 / (rho * cv)
//!         f = 1 / (1 + betaf * opacP * c * dt)
//!         NOTE: if J = opacP * c * aR * T^4,
//!               then f = 1 / (1 + 4 * J * dt / (rho * cv * T))
TaskStatus UpdateDerivedTransportFields(MeshData<Real> *md, const Real dt) {
  PARTHENON_INSTRUMENT
  auto pm = md->GetParentPointer();
  auto &jbn = pm->packages.Get("jaybenne");
  auto &frequency_type = jbn->template Param<FrequencyType>("frequency_type");
  if (frequency_type == FrequencyType::gray) {
    return UpdateDerivedTransportFieldsImpl<FrequencyType::gray>(md, dt);
  } else if (frequency_type == FrequencyType::multigroup) {
    return UpdateDerivedTransportFieldsImpl<FrequencyType::multigroup>(md, dt);
  } else {
    PARTHENON_FAIL("frequency_type not recognized!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus TaskStatus DefragParticles(MeshBlock *pmb)
//! \brief  NOTE(PDM): currently unused???
//! TODO(BRR) We should re-enable this but add a runtime parameter that sets the
//! fractional fragmentation of the memory pool above which we defragment.
TaskStatus DefragParticles(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  auto pm = md->GetParentPointer();
  auto &jbn = pm->packages.Get("jaybenne");
  auto &min_swarm_occupancy = jbn->template Param<Real>("min_swarm_occupancy");
  const int nblocks = md->NumBlocks();

  for (int b = 0; b <= nblocks - 1; ++b) {
    auto &swarm = md->GetSwarmData(b)->Get(photons_swarm_name);
    if (swarm->GetNumActive() > 0) {
      if (swarm->GetPackingEfficiency() < min_swarm_occupancy) {
        swarm->Defrag();
      }
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus EvaluateRadiationEnergy
//! \brief
template <typename T>
TaskStatus EvaluateRadiationEnergy(T *md) {
  PARTHENON_INSTRUMENT
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
          const int &ip = ppack_i(b, ph::ijk(0), n);
          const int &jp = ppack_i(b, ph::ijk(1), n);
          const int &kp = ppack_i(b, ph::ijk(2), n);
          const Real &dv = coords.CellVolume(kp, jp, ip);
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
  PARTHENON_INSTRUMENT
  auto &jb_pkg = mbd->GetBlockPointer()->packages.Get("jaybenne");
  const auto &fd = jb_pkg->template Param<FrequencyType>("frequency_type");

  if (is_thermal) {
    if (fd == FrequencyType::gray) {
      jaybenne::SourcePhotons<MeshBlockData<Real>, jaybenne::SourceType::thermal,
                              FrequencyType::gray>(mbd, 0.0, 0.0);
    } else if (fd == FrequencyType::multigroup) {
      jaybenne::SourcePhotons<MeshBlockData<Real>, jaybenne::SourceType::thermal,
                              FrequencyType::multigroup>(mbd, 0.0, 0.0);
    }
  }

  // Call this so radiation field variables are properly initialized for outputs
  jaybenne::EvaluateRadiationEnergy<MeshBlockData<Real>>(mbd);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus UpdateFluid
//! \brief
TaskStatus UpdateFluid(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  namespace fj = field::jaybenne;
  namespace fjh = field::jaybenne::host;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jb_pkg = pm->packages.Get("jaybenne");
  if (!(jb_pkg->template Param<bool>("do_feedback"))) return TaskStatus::complete;

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
        const Real &dv = coords.CellVolume(k, j, i);
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
