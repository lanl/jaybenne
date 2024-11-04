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
                         Metadata::ForceRemeshComm});
  pkg->AddField(field::material::density::name(), m);
  pkg->AddField(field::material::internal_energy::name(), m);

  // Volumetric internal energy and specific internal energy
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(field::material::sie::name(), m);

  // Equation of state
  const Real gamma = pin->GetOrAddReal("mcblock", "gamma", 1.66666666667);
  const Real cv = pin->GetOrAddReal("mcblock", "cv", 1. / (gamma - 1.));
  EOS eos_h = singularity::IdealGas(gamma - 1., cv); // TODO(BRR) gamma = 5/3?
  pkg->AddParam<>("eos_h", eos_h);
  pkg->AddParam<>("eos_d", eos_h.GetOnDevice());

  // Absorption opacity model
  OpacityModel opacity_model;
  Opacity opacity;
  std::string opacity_model_name = pin->GetString("mcblock", "opacity_model");
  if (opacity_model_name == "none") {
    opacity_model = OpacityModel::none;
    opacity = singularity::photons::Gray(0.0);
  } else if (opacity_model_name == "constant") {
    opacity_model = OpacityModel::constant;
    Real kappa = pin->GetReal("mcblock", "opacity_constant_value");
    opacity = singularity::photons::Gray(kappa);
  } else {
    // nothing else supported for now
    PARTHENON_FAIL("Only none or constant opacity models supported!");
  }
  pkg->AddParam<>("opacity_model", opacity_model);
  pkg->AddParam<>("opacity_h", opacity);

  // Scattering opacity model
  ScatteringModel scattering_model;
  Scattering scattering;
  std::string scattering_model_name =
      pin->GetOrAddString("mcblock", "scattering_model", "none");
  if (scattering_model_name == "none") {
    scattering_model = ScatteringModel::none;
    scattering = singularity::photons::GrayS(0.0, 1.0);
  } else if (scattering_model_name == "constant") {
    // TODO(BRR): parse 2nd argument: average particle mass?
    scattering_model = ScatteringModel::constant;
    Real kappa_s = pin->GetReal("mcblock", "scattering_constant_value");
    scattering = singularity::photons::GrayS(kappa_s, 1.0);
  } else {
    PARTHENON_FAIL("Only none or constant scattering models supported!");
  }
  pkg->AddParam<>("scattering_model", scattering_model);
  pkg->AddParam<>("scattering_h", scattering);

  pkg->PreFillDerivedMesh = UpdateDerived;

  return pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  void ProblemGenerator
//! \brief Generate initial conditions for problems
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  namespace fm = field::material;

  auto mbd = pmb->meshblock_data.Get().get();
  auto &resolved_pkgs = pmb->resolved_packages;

  auto &mcb = pmb->packages.Get("mcblock");
  const Real &rho0 = mcb->Param<Real>("initial_density");
  const Real &tt0 = mcb->Param<Real>("initial_temperature");
  const auto &initial_radiation = mcb->Param<InitialRadiation>("initial_radiation");
  auto eos = mcb->Param<EOS>("eos_d");

  // Create SparsePack
  static auto desc = MakePackDescriptor<fm::density, fm::sie>(resolved_pkgs.get());
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

  const auto problem_id = mcb->Param<std::string>("problem_id");
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

  jaybenne::InitializeRadiation(mbd, (initial_radiation == InitialRadiation::thermal));
}

//----------------------------------------------------------------------------------------
//! \fn void UpdateDerived
//! \brief Updates Mcblock derived variables following a Jaybenne step
void UpdateDerived(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<field::material::density, field::material::internal_energy,
                         field::material::sie>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdateDerived", parthenon::DevExecSpace(), 0,
      vmesh.GetNBlocks() - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Set derived fields
        Real &dd = vmesh(b, field::material::density(), k, j, i);
        Real &uu = vmesh(b, field::material::internal_energy(), k, j, i);
        Real &sie = vmesh(b, field::material::sie(), k, j, i);
        sie = uu / dd;
      });
}

//----------------------------------------------------------------------------------------
//! \fn void PostInitialization
//! \brief
void PostInitialization(MeshBlock *pmb, ParameterInput *pin) {
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
  auto eos_h = mcblock->Param<EOS>("eos_h");
  auto opacity_h = mcblock->Param<Opacity>("opacity_h");
  auto scattering_h = mcblock->Param<Scattering>("scattering_h");
  packages.Add(jaybenne::Initialize(pin.get(), opacity_h, scattering_h, eos_h));
  return packages;
}

} // namespace mcblock
