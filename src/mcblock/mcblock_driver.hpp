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
#ifndef MCBLOCK_MCBLOCK_DRIVER_HPP_
#define MCBLOCK_MCBLOCK_DRIVER_HPP_

// Jaybenne includes
#include "../jaybenne/jaybenne.hpp"

// Mcblock includes
#include "mcblock.hpp"

namespace mcblock {

//----------------------------------------------------------------------------------------
//! \class  McblockDriver
//! \brief McblockDriver is the driver class for the Jaybenne application MCBLOCK that
//! uses the Jaybenne package for thermal radiative transport and the MCBLOCK package for
//! a material energy grid. McblockDriver inherits from Parthenon's
//! MultiStageBlockTaskDriver to create a custom driver class for thermal radiation
//! transport. This class creates the task list for this application.
class McblockDriver : public EvolutionDriver {
 public:
  using Integrator_t = parthenon::LowStorageIntegrator;

  McblockDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm);

  TaskListStatus Step();

  TaskCollection HostUpdateTasks();

 private:
  std::unique_ptr<Integrator_t> integrator;
  StateDescriptor *jb_pkg;
};

} // namespace mcblock

#endif // MCBLOCK_MCBLOCK_DRIVER_HPP_
