//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
#ifndef JAYBENNE_BOUNDARIES_HPP_
#define JAYBENNE_BOUNDARIES_HPP_

// Jaybenne includes
#include "jaybenne_variables.hpp"

namespace jaybenne {

//----------------------------------------------------------------------------------------
//! \fn  PhotonReflectBC
//! \brief Reflecting boundary conditions for all boundaries for photons
template <BoundaryFace BFACE>
void PhotonReflectBC(std::shared_ptr<Swarm> &swarm) {

  auto swarm_d_ = swarm->GetDeviceContext();
  int max_active_index = swarm->GetMaxActiveIndex();

  // Get relevant fields
  auto &x_ = swarm->Get<Real>(swarm_position::x::name()).Get();
  auto &y_ = swarm->Get<Real>(swarm_position::y::name()).Get();
  auto &z_ = swarm->Get<Real>(swarm_position::z::name()).Get();
  auto &v_ = swarm->Get<Real>(particle::photons::v::name()).Get();
  auto &ijk_ = swarm->Get<int>(particle::photons::ijk::name()).Get();
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        // These redefinitons are required to avoid a compilation error from CUDA about
        // first-capture occuring inside an if consexpr
        auto &x = x_;
        auto &y = y_;
        auto &z = z_;
        auto &v = v_;
        auto &ijk = ijk_;
        auto &swarm_d = swarm_d_;
        if constexpr (BFACE == BoundaryFace::inner_x1) {
          if (x(n) < swarm_d.x_min_global_) {
            x(n) = swarm_d.x_min_global_ + (swarm_d.x_min_global_ - x(n));
            v(0, n) = -v(0, n);
            swarm_d.Xtoijk(x(n), y(n), z(n), ijk(0, n), ijk(1, n), ijk(2, n));
          }
        } else if constexpr (BFACE == BoundaryFace::outer_x1) {
          if (x(n) > swarm_d.x_max_global_) {
            x(n) = swarm_d.x_max_global_ - (x(n) - swarm_d.x_max_global_);
            v(0, n) = -v(0, n);
            swarm_d.Xtoijk(x(n), y(n), z(n), ijk(0, n), ijk(1, n), ijk(2, n));
          }
        } else if constexpr (BFACE == BoundaryFace::inner_x2) {
          if (y(n) < swarm_d.y_min_global_) {
            y(n) = swarm_d.y_min_global_ + (swarm_d.y_min_global_ - y(n));
            v(1, n) = -v(1, n);
            swarm_d.Xtoijk(x(n), y(n), z(n), ijk(0, n), ijk(1, n), ijk(2, n));
          }
        } else if constexpr (BFACE == BoundaryFace::outer_x2) {
          if (y(n) > swarm_d.y_max_global_) {
            y(n) = swarm_d.y_max_global_ - (y(n) - swarm_d.y_max_global_);
            v(1, n) = -v(1, n);
            swarm_d.Xtoijk(x(n), y(n), z(n), ijk(0, n), ijk(1, n), ijk(2, n));
          }
        } else if constexpr (BFACE == BoundaryFace::inner_x3) {
          if (z(n) < swarm_d.z_min_global_) {
            z(n) = swarm_d.z_min_global_ + (swarm_d.z_min_global_ - z(n));
            v(2, n) = -v(2, n);
            swarm_d.Xtoijk(x(n), y(n), z(n), ijk(0, n), ijk(1, n), ijk(2, n));
          }
        } else if constexpr (BFACE == BoundaryFace::outer_x3) {
          if (z(n) > swarm_d.z_max_global_) {
            z(n) = swarm_d.z_max_global_ - (z(n) - swarm_d.z_max_global_);
            v(2, n) = -v(2, n);
            swarm_d.Xtoijk(x(n), y(n), z(n), ijk(0, n), ijk(1, n), ijk(2, n));
          }
        }
      });
}

} // namespace jaybenne

#endif // JAYBENNE_BOUNDARIES_HPP_
