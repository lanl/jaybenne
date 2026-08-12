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

// C/C++ includes
#include <string>

// Parthenon includes
#include <parthenon_manager.hpp>

// Jaybenne includes
#include "../jaybenne/jaybenne.hpp"

// Mcblock includes
#include "mcblock.hpp"

namespace mcblock {

void SwarmBoundaryDoNothing(std::shared_ptr<Swarm> &) {}
void UpdateDerived(MeshData<Real> *md);

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor MCBlock::Initialize
//! \brief Initialize the MCBlock physics package
//! This function defines and sets the parameters associated with MCBlock, and enrolls the
//! data variables associated with this physics package.
//! pin: Runtime input parameters
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("mcblock");

  PARTHENON_REQUIRE(pin->GetString("parthenon/time", "integrator") == "rk1",
                    "McBlock driver only supports first order time integration");

  // Problem name
  auto problem_id = pin->GetString("parthenon/job", "problem_id");
  pkg->AddParam<>("problem_id", problem_id);

  // Initial temperature (K)
  Real initial_temperature = pin->GetReal("mcblock", "initial_temperature");
  pkg->AddParam<>("initial_temperature", initial_temperature);

  // Initial density (g cm^-3)
  Real initial_density = pin->GetReal("mcblock", "initial_density");
  pkg->AddParam<>("initial_density", initial_density);

  // Initial radiation (none, thermal)
  std::string initial_radiation = pin->GetString("mcblock", "initial_radiation");
  InitialRadiation initial_radiation_val;
  if (initial_radiation == "none") {
    initial_radiation_val = InitialRadiation::none;
  } else if (initial_radiation == "thermal") {
    initial_radiation_val = InitialRadiation::thermal;
  } else {
    PARTHENON_FAIL("Only none or thermal initial radiation supported!");
  }
  pkg->AddParam<>("initial_radiation", initial_radiation_val);

  // Density and total energy fields
  Metadata m = Metadata({Metadata::Cell, Metadata::FillGhost, Metadata::OneCopy,
                         Metadata::ForceRemeshComm, Metadata::Restart});
  pkg->AddField(field::material::density::name(), m);
  pkg->AddField(field::material::internal_energy::name(), m);

  // opacity fields
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(field::material::absorption_opacity::name(), m);
  pkg->AddField(field::material::scattering_opacity::name(), m);

  // Volumetric internal energy and specific internal energy
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Restart});
  pkg->AddField(field::material::sie::name(), m);

  // Equation of state
  const Real gamma = pin->GetOrAddReal("mcblock", "gamma", 1.66666666667);
  const Real cv = pin->GetOrAddReal("mcblock", "cv", 1. / (gamma - 1.));
  EOS eos_h = singularity::IdealGas(gamma - 1., cv); // TODO(BRR) gamma = 5/3?
  pkg->AddParam<>("eos_h", eos_h);
  pkg->AddParam<>("eos_d", eos_h.GetOnDevice());

  // Units (conversion factors from code to CGS)
  const Real time_scale = pin->GetOrAddReal("mcblock", "time_scale", 1.);
  const Real mass_scale = pin->GetOrAddReal("mcblock", "mass_scale", 1.);
  const Real length_scale = pin->GetOrAddReal("mcblock", "length_scale", 1.);
  const Real temperature_scale = pin->GetOrAddReal("mcblock", "temperature_scale", 1.);
  pkg->AddParam<>("time_scale", time_scale);
  pkg->AddParam<>("mass_scale", mass_scale);
  pkg->AddParam<>("length_scale", length_scale);
  pkg->AddParam<>("temperature_scale", temperature_scale);

  // Frequency discretization
  std::string frequency_type_name = pin->GetString("mcblock", "frequency_type");
  FrequencyType frequency_type;
  if (frequency_type_name == "gray") {
    frequency_type = FrequencyType::gray;
  } else if (frequency_type_name == "multigroup") {
    frequency_type = FrequencyType::multigroup;
  } else {
    PARTHENON_FAIL("\"mcblock/frequency_type\" not recognized!");
  }

  // set opacity group bounds: gray goes from 0 to infty
  std::vector<Real> opac_grp_bnds = {0.0, std::numeric_limits<Real>::infinity()};
  // reset to parsed input (for non-tabular opacity models) for multigroup
  if (frequency_type == FrequencyType::multigroup) {
    const Real numin = pin->GetReal("jaybenne", "numin"); // in Hz
    const Real numax = pin->GetReal("jaybenne", "numax"); // in Hz
    const int n_nubins = pin->GetInteger("jaybenne", "n_nubins");
    // reset opacity group bounds
    // NOTE: these are group edges, not interior points
    opac_grp_bnds.assign(n_nubins + 1, 0.0);
    // assume uniform log-spacing, grid is midpoints in log-space
    const Real dlnu = (std::log(numax) - std::log(numin)) / n_nubins;
    for (int n = 0; n < n_nubins + 1; ++n) {
      opac_grp_bnds[n] = numin * std::exp(n * dlnu);
    }
  }
  const int NG = static_cast<int>(opac_grp_bnds.size()) - 1;

  // Absorption opacity model
  MeanOpacity mopacity;
  std::string abs_model = pin->GetString("mcblock/absorption", "opacity_model");

  if (abs_model == "table") {
    // table read from file
    std::string table_filename = pin->GetString("mcblock/absorption", "opacity_table");
    mopacity =
        singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
            singularity::photons::MeanOpacityBase(table_filename), time_scale, mass_scale,
            length_scale, temperature_scale);
  } else if (abs_model == "none") {
    // none = 0 absorption
    auto model = singularity::photons::Gray(0.0);
    mopacity =
        singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
            singularity::photons::MeanOpacityBase(model, -1., 1., 2, -1., 1., 2,
                                                  opac_grp_bnds, NG),
            time_scale, mass_scale, length_scale, temperature_scale);
  } else if (abs_model == "constant") {
    Real kappa = pin->GetReal("mcblock/absorption", "constant_value");
    auto model = singularity::photons::Gray(kappa);
    mopacity =
        singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
            singularity::photons::MeanOpacityBase(model, -1., 1., 2, -1., 1., 2,
                                                  opac_grp_bnds, NG),
            time_scale, mass_scale, length_scale, temperature_scale);
  } else if (abs_model == "powerlaw") {
    // NOTE: reference values (ref) and offsets (off) must always be in cgs units
    const Real kappa0 = pin->GetReal("mcblock/absorption", "kappa0");
    const Real rho_exp = pin->GetReal("mcblock/absorption", "rho_exp");
    const Real temp_exp = pin->GetReal("mcblock/absorption", "temp_exp");
    const Real nu_exp = pin->GetReal("mcblock/absorption", "nu_exp");
    const Real nu_ref = pin->GetReal("mcblock/absorption", "nu_ref");
    const Real nu_off = pin->GetOrAddReal("mcblock/absorption", "nu_off", 0.0);
    const Real rho_ref = pin->GetReal("mcblock/absorption", "rho_ref");
    const Real rho_off = pin->GetOrAddReal("mcblock/absorption", "rho_off", 0.0);
    const Real temp_ref = pin->GetReal("mcblock/absorption", "temp_ref");
    const Real temp_off = pin->GetOrAddReal("mcblock/absorption", "temp_off", 0.0);
    const bool do_stim_emit =
        pin->GetOrAddBoolean("mcblock/absorption", "do_stim_emit", false);
    auto model = singularity::photons::PowerLaw(kappa0, rho_exp, temp_exp, nu_exp, nu_ref,
                                                nu_off, rho_ref, rho_off, temp_ref,
                                                temp_off, do_stim_emit);
    mopacity =
        singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
            singularity::photons::MeanOpacityBase(model, -1., 1., 2, -1., 1., 2,
                                                  opac_grp_bnds, NG),
            time_scale, mass_scale, length_scale, temperature_scale);
  } else if (abs_model == "ep_bremss") {
    auto model = singularity::photons::EPBremss();
    mopacity =
        singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
            singularity::photons::MeanOpacityBase(model, -1., 1., 2, -1., 1., 2,
                                                  opac_grp_bnds, NG),
            time_scale, mass_scale, length_scale, temperature_scale);
  } else {
    // nothing else supported for now
    PARTHENON_FAIL("Invalid absorption opacity model selected!");
  }

  pkg->AddParam<>("frequency_type", frequency_type);
  pkg->AddParam<>("mopacity_h", mopacity);

  // Scattering opacity model
  // TODO(BRR) Remove apm with switch in singularity-opac to cm^2/g opacities?
  const Real apm =
      pin->GetOrAddReal("mcblock", "apm", 1.); // Average particle mass (code units)
  MeanScattering mscattering;
  std::string sct_model =
      pin->GetOrAddString("mcblock/scattering", "opacity_model", "none");

  if (sct_model == "none") {
    auto smodel = singularity::photons::GrayS(0.0, apm);
    mscattering =
        singularity::photons::MeanNonCGSUnitsS<singularity::photons::MeanSOpacityBase>(
            singularity::photons::MeanSOpacityBase(smodel, -1., 1., 2, -1., 1., 2,
                                                   opac_grp_bnds, NG),
            time_scale, mass_scale, length_scale, temperature_scale);
  } else if (sct_model == "constant") {
    Real kappa_s = pin->GetReal("mcblock/scattering", "constant_value");
    auto smodel = singularity::photons::GrayS(kappa_s, apm);
    mscattering =
        singularity::photons::MeanNonCGSUnitsS<singularity::photons::MeanSOpacityBase>(
            singularity::photons::MeanSOpacityBase(smodel, -1., 1., 2, -1., 1., 2,
                                                   opac_grp_bnds, NG),
            time_scale, mass_scale, length_scale, temperature_scale);
  } else if (sct_model == "powerlaw") {
    // NOTE: reference values (ref) and offsets (off) must always be in cgs units
    const Real kappa0 = pin->GetReal("mcblock/scattering", "kappa0");
    const Real rho_exp = pin->GetReal("mcblock/scattering", "rho_exp");
    const Real temp_exp = pin->GetReal("mcblock/scattering", "temp_exp");
    const Real nu_exp = pin->GetReal("mcblock/scattering", "nu_exp");
    const Real nu_ref = pin->GetReal("mcblock/scattering", "nu_ref");
    const Real nu_off = pin->GetOrAddReal("mcblock/scattering", "nu_off", 0.0);
    const Real rho_ref = pin->GetReal("mcblock/scattering", "rho_ref");
    const Real rho_off = pin->GetOrAddReal("mcblock/scattering", "rho_off", 0.0);
    const Real temp_ref = pin->GetReal("mcblock/scattering", "temp_ref");
    const Real temp_off = pin->GetOrAddReal("mcblock/scattering", "temp_off", 0.0);
    auto smodel =
        singularity::photons::PowerLawS(kappa0, rho_exp, temp_exp, nu_exp, nu_ref, nu_off,
                                        rho_ref, rho_off, temp_ref, temp_off);
    mscattering =
        singularity::photons::MeanNonCGSUnitsS<singularity::photons::MeanSOpacityBase>(
            singularity::photons::MeanSOpacityBase(smodel, -1., 1., 2, -1., 1., 2,
                                                   opac_grp_bnds, NG),
            time_scale, mass_scale, length_scale, temperature_scale);
  } else {
    PARTHENON_FAIL("Invalid scattering opacity model selected!");
  }

  pkg->AddParam<>("mscattering_h", mscattering);

  pkg->PreFillDerivedMesh = UpdateDerived;

  return pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  void ProblemGenerator
//! \brief Generate initial conditions for problems
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  namespace fm = field::material;
  using singularity::photons::OpacityAveraging;
  using singularity::photons::Planck;
  using singularity::photons::Rosseland;

  auto mbd = pmb->meshblock_data.Get().get();
  auto &resolved_pkgs = pmb->resolved_packages;

  auto &mcb = pmb->packages.Get("mcblock");
  // TODO: move device opacity objects from jaybenne to mcblock
  auto &jbn = pmb->packages.Get("jaybenne");
  const auto &frequency_type = mcb->Param<FrequencyType>("frequency_type");
  const Real &rho0 = mcb->template Param<Real>("initial_density");
  const Real &tt0 = mcb->template Param<Real>("initial_temperature");
  const auto &initial_radiation =
      mcb->template Param<InitialRadiation>("initial_radiation");
  auto eos = mcb->template Param<EOS>("eos_d");
  MeanOpacity mopacity;
  MeanScattering mscattering;
  if (frequency_type == FrequencyType::gray) {
    mopacity = jbn->template Param<MeanOpacity>("mopacity_d");
    mscattering = jbn->template Param<MeanScattering>("mscattering_d");
  }

  // Create SparsePack
  static auto desc = MakePackDescriptor<fm::density, fm::sie, fm::absorption_opacity,
                                        fm::scattering_opacity>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(mbd);

  // Indexing and dimensionality
  const auto &ib = mbd->GetBoundsI(IndexDomain::interior);
  const auto &jb = mbd->GetBoundsJ(IndexDomain::interior);
  const auto &kb = mbd->GetBoundsK(IndexDomain::interior);
  const int &nblocks = vmesh.GetNBlocks();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Initialize rho, T, u", parthenon::DevExecSpace(), 0,
      nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        const Real cv = eos.SpecificHeatFromDensityInternalEnergy(rho0, 1.0);
        vmesh(b, fm::density(), k, j, i) = rho0;
        vmesh(b, fm::sie(), k, j, i) = cv * tt0;
      });

  const auto problem_id = mcb->template Param<std::string>("problem_id");
  if (problem_id == "stepdiff") {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Initialize stepdiff", parthenon::DevExecSpace(), 0,
        nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          auto &coords = vmesh.GetCoordinates(b);
          const Real x1v = coords.template Xc<parthenon::X1DIR>(i);
          if (x1v >= 0.0) {
            const Real ttlow = 1.0e-5 * tt0;
            const Real cv = eos.SpecificHeatFromDensityInternalEnergy(rho0, 1.0);
            vmesh(b, fm::sie(), k, j, i) = cv * ttlow;
          }
        });
  }

  // initialize opacity (TODO: only gray for now)
  if (frequency_type == FrequencyType::gray) {

    // get opacity average indicators
    const auto &use_planck = jbn->template Param<bool>("use_planck");
    const auto &use_rosseland = jbn->template Param<bool>("use_rosseland");
    // set opacity mode for grey opacity
    const OpacityAveraging gmode = (use_planck && !use_rosseland) ? Planck : Rosseland;

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Initialize opacity", parthenon::DevExecSpace(), 0,
        nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          const Real &rho = vmesh(b, fm::density(), k, j, i);
          const Real &sie = vmesh(b, fm::sie(), k, j, i);
          const Real temp = eos.TemperatureFromDensityInternalEnergy(rho, sie);
          const Real aa = mopacity.AbsorptionCoefficient(rho, temp, 0, gmode);
          const Real ss = mscattering.ScatteringCoefficient(rho, temp, 0, gmode);
          vmesh(b, fm::absorption_opacity(), k, j, i) = aa;
          vmesh(b, fm::scattering_opacity(), k, j, i) = ss;
        });
  }

  jaybenne::InitializeRadiation(mbd, (initial_radiation == InitialRadiation::thermal));
}

//----------------------------------------------------------------------------------------
//! \fn void UpdateDerived
//! \brief Updates Mcblock derived variables following a Jaybenne step
void UpdateDerived(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  namespace fm = field::material;
  using parthenon::MakePackDescriptor;
  using singularity::photons::OpacityAveraging;
  using singularity::photons::Planck;
  using singularity::photons::Rosseland;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &jbn = pm->packages.Get("jaybenne");
  auto &mcb = pm->packages.Get("mcblock");
  auto eos = mcb->template Param<EOS>("eos_d");

  const auto &frequency_type = mcb->template Param<FrequencyType>("frequency_type");

  MeanOpacity mopacity;
  MeanScattering mscattering;
  if (frequency_type == FrequencyType::gray) {
    mopacity = jbn->template Param<MeanOpacity>("mopacity_d");
    mscattering = jbn->template Param<MeanScattering>("mscattering_d");
  }

  // Packing and indexing
  static auto desc = MakePackDescriptor<fm::density, fm::internal_energy, fm::sie,
                                        fm::absorption_opacity, fm::scattering_opacity>(
      resolved_pkgs.get());

  auto vmesh = desc.GetPack(md);
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdateDerived", parthenon::DevExecSpace(), 0,
      vmesh.GetNBlocks() - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Set derived fields
        Real &dd = vmesh(b, fm::density(), k, j, i);
        Real &uu = vmesh(b, fm::internal_energy(), k, j, i);
        Real &sie = vmesh(b, fm::sie(), k, j, i);
        sie = uu / dd;
      });

  // update opacity (TODO: only gray for now)
  if (frequency_type == FrequencyType::gray) {

    // get opacity average indicators
    const auto &use_planck = jbn->template Param<bool>("use_planck");
    const auto &use_rosseland = jbn->template Param<bool>("use_rosseland");
    // set opacity mode for grey opacity
    const OpacityAveraging gmode = (use_planck && !use_rosseland) ? Planck : Rosseland;

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Update opacity", parthenon::DevExecSpace(), 0,
        vmesh.GetNBlocks() - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          const Real &rho = vmesh(b, fm::density(), k, j, i);
          const Real &sie = vmesh(b, fm::sie(), k, j, i);
          const Real temp = eos.TemperatureFromDensityInternalEnergy(rho, sie);
          const Real aa = mopacity.AbsorptionCoefficient(rho, temp, 0, gmode);
          const Real ss = mscattering.ScatteringCoefficient(rho, temp, 0, gmode);
          vmesh(b, fm::absorption_opacity(), k, j, i) = aa;
          vmesh(b, fm::scattering_opacity(), k, j, i) = ss;
        });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PostInitialization
//! \brief
void PostInitialization(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  auto md = pmb->meshblock_data.Get().get();
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<field::material::density, field::material::internal_energy,
                         field::material::sie>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PostInitialization", parthenon::DevExecSpace(), 0,
      vmesh.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Set internal energy
        Real &dd = vmesh(b, field::material::density(), k, j, i);
        Real &uu = vmesh(b, field::material::internal_energy(), k, j, i);
        Real &sie = vmesh(b, field::material::sie(), k, j, i);
        uu = dd * sie;
      });
}

//----------------------------------------------------------------------------------------
//! \fn  void ProblemModifier
//! \brief
void ProblemModifier(parthenon::ParthenonManager *pman) {
  using BF = parthenon::BoundaryFace;

  // Register custom boundary conditions
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::inner_x1, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::inner_x1>);
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::outer_x1, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::outer_x1>);
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::inner_x2, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::inner_x2>);
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::outer_x2, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::outer_x2>);
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::inner_x3, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::inner_x3>);
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::outer_x3, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::outer_x3>);
}

//----------------------------------------------------------------------------------------
//! \fn  Packages_t ProcessPackages
//! \brief Enroll all packages used by this application
Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(mcblock::Initialize(pin.get()));
  auto &mcblock = packages.Get("mcblock");
  auto eos_h = mcblock->template Param<EOS>("eos_h");
  auto mopacity_h = mcblock->template Param<MeanOpacity>("mopacity_h");
  auto mscattering_h = mcblock->template Param<MeanScattering>("mscattering_h");
  packages.Add(jaybenne::Initialize(pin.get(), mopacity_h, mscattering_h, eos_h));
  return packages;
}

} // namespace mcblock
