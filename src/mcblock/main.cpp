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
#include <stdio.h>

// Mcblock includes
#include "mcblock.hpp"
#include "mcblock_driver.hpp"

int main(int argc, char *argv[]) {
  parthenon::ParthenonManager pman;

  // initialize MPI and Kokkos, parse the input deck, and set up Parthenon
  auto parthenon_status = pman.ParthenonInitEnv(argc, argv);
  if (parthenon_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (parthenon_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  // Redefine parthenon defaults
  pman.app_input->ProcessPackages = mcblock::ProcessPackages;
  mcblock::ProblemModifier(&pman);
  pman.app_input->ProblemGenerator = mcblock::ProblemGenerator;
  pman.app_input->PostInitialization = mcblock::PostInitialization;

  // Call ParthenonInit to set up the mesh
  pman.ParthenonInitPackagesAndMesh();

  // Initialize the driver
  mcblock::McblockDriver driver(pman.pinput.get(), pman.app_input.get(),
                                pman.pmesh.get());

  // Run the simulation
  auto driver_status = driver.Execute();

  // Finalize Parthenon (and MPI and Kokkos)
  pman.ParthenonFinalize();

  return 0;
}
