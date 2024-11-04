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
#ifndef JAYBENNE_JAYBENNE_VARIABLES_HPP_
#define JAYBENNE_JAYBENNE_VARIABLES_HPP_

// C++ includes
#include <string>

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

static const std::string photons_swarm_name = "photons";

#define JAYBENNE_FIELD_VARIABLE(ns, varname)                                             \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace field {
namespace jaybenne {
JAYBENNE_FIELD_VARIABLE(field.jaybenne, energy_tally);
JAYBENNE_FIELD_VARIABLE(field.jaybenne, fleck_factor);
JAYBENNE_FIELD_VARIABLE(field.jaybenne, ddmc_face_prob);
JAYBENNE_FIELD_VARIABLE(field.jaybenne, source_ew_per_cell);
JAYBENNE_FIELD_VARIABLE(field.jaybenne, source_num_per_cell);
JAYBENNE_FIELD_VARIABLE(field.jaybenne, energy_delta);
namespace host {
typedef HOST_DENSITY density;
typedef HOST_SPECIFIC_INTERNAL_ENERGY sie;
typedef HOST_UPDATE_ENERGY update_energy;
} // namespace host
} // namespace jaybenne
} // namespace field

namespace particle {
namespace photons {
SWARM_VARIABLE(Real, particle.photons, time);
SWARM_VARIABLE(Real, particle.photons, weight);
SWARM_VARIABLE(Real, particle.photons, energy);
SWARM_VARIABLE(Real, particle.photons, v);
SWARM_VARIABLE(int, particle.photons, ijk);
} // namespace photons
} // namespace particle

#endif // JAYBENNE_JAYBENNE_VARIABLES_HPP_
