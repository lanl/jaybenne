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

// Mcblock includes
#include "mcblock_driver.hpp"

namespace mcblock {

McblockDriver::McblockDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : EvolutionDriver(pin, app_in, pm), integrator(std::make_unique<Integrator_t>(pin)) {

  // Sanity checks for input file
  pin->CheckRequired("parthenon/mesh", "ix1_bc");
  pin->CheckRequired("parthenon/mesh", "ox1_bc");
  pin->CheckRequired("parthenon/mesh", "ix2_bc");
  pin->CheckRequired("parthenon/mesh", "ox2_bc");
  pin->CheckRequired("parthenon/mesh", "ix3_bc");
  pin->CheckRequired("parthenon/mesh", "ox3_bc");

  // Jaybenne package
  jb_pkg = pm->packages.Get("jaybenne").get();
}

//----------------------------------------------------------------------------------------
//! \fn  TaskListStatus McblockDriver::Step
//! \brief Defines the task ordering for an integrator timestep.
//!        Overrides the parthenon EvolutionDriver virtual method.
TaskListStatus McblockDriver::Step() {

  // Set numerical timestep
  // Particles are always updated in a first order sense; RK substeps are ignored.
  integrator->dt = tm.dt;
  const Real &dt = integrator->dt;

  // One cycle of radiation transport
  auto status = jaybenne::RadiationStep(pmesh, this->tm.time, integrator->dt).Execute();
  if (status != TaskListStatus::complete) return status;

  // compute new dt
  status = HostUpdateTasks().Execute();

  return status;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection McblockDriver::HostUpdateTasks
//! \brief Update host material properties after radiation transport step.
TaskCollection McblockDriver::HostUpdateTasks() {
  using namespace ::parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  const int num_partitions = pmesh->DefaultNumPartitions();
  auto &post_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = post_region[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    auto bcs = parthenon::AddBoundaryExchangeTasks(none, tl, base, pmesh->multilevel);
    auto derived = tl.AddTask(bcs, FillDerived<MeshData<Real>>, base.get());
    auto new_dt = tl.AddTask(derived, EstimateTimestep<MeshData<Real>>, base.get());
  }

  return tc;
}

} // namespace mcblock
