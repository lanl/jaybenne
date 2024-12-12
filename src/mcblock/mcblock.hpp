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
#ifndef MCBLOCK_MCBLOCK_HPP_
#define MCBLOCK_MCBLOCK_HPP_

// Mcblock includes
#include "mcblock.hpp"

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon;
using namespace parthenon::constants;
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

namespace mcblock {

// Model enums
enum class InitialRadiation { none, thermal };
enum class OpacityModel { none, constant, epbremss };
enum class ScatteringModel { none, constant };

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
void ProblemModifier(parthenon::ParthenonManager *pman);
Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);
void UpdateDerived(MeshData<Real> *md);
void PostInitialization(MeshBlock *pmb, ParameterInput *pin);

} // namespace mcblock

#endif // MCBLOCK_MCBLOCK_HPP_
