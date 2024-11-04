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
#ifndef MCBLOCK_MCBLOCK_VARIABLES_HPP_
#define MCBLOCK_MCBLOCK_VARIABLES_HPP_

// C++ includes
#include <string>

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#define MCBLOCK_FIELD_VARIABLE(ns, varname)                                              \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace field {
namespace material {
MCBLOCK_FIELD_VARIABLE(field.material, density);
MCBLOCK_FIELD_VARIABLE(field.material, sie);
MCBLOCK_FIELD_VARIABLE(field.material, internal_energy);
} // namespace material
} // namespace field

#endif // MCBLOCK_MCBLOCK_VARIABLES_HPP_
