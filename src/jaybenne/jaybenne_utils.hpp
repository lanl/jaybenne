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
#ifndef JAYBENNE_JAYBENNE_UTILS_HPP_
#define JAYBENNE_JAYBENNE_UTILS_HPP_

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace jaybenne {

template <typename Function, typename T>
KOKKOS_FORCEINLINE_FUNCTION void
par_reduce_inner(team_mbr_t team_member, const int kl, const int ku, const int jl,
                 const int ju, const int il, const int iu, const Function &function,
                 T reduction) {
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NkNjNi = Nk * Nj * Ni;
  const int NjNi = Nj * Ni;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, NkNjNi),
      [&](const int &idx, int &lreduce) {
        int k = idx / NjNi;
        int j = (idx - k * NjNi) / Ni;
        int i = idx - k * NjNi - j * Ni;
        k += kl;
        j += jl;
        i += il;
        function(k, j, i, lreduce);
      },
      reduction);
}

template <typename ExecSpace, typename Function, typename U>
void global_sum_reduce(const std::string &label, const ExecSpace &space,
                       const int nblocks, const int kl, const int ku, const int jl,
                       const int ju, const int il, const int iu, const Function &function,
                       U &globally_reduced, const bool allreduce = false) {
  PARTHENON_INSTRUMENT
  PARTHENON_DEBUG_REQUIRE(std::is_scalar<U>::value,
                          "global_sum_reduce only works on scalars.");

  // reduce over local blocks
  U totag = 0;
  parthenon::par_reduce(parthenon::loop_pattern_mdrange_tag, label, space, 0, nblocks - 1,
                        kl, ku, jl, ju, il, iu, function, Kokkos::Sum<U>(totag));
  Kokkos::fence();

  // reduce over MPI ranks
#ifdef MPI_PARALLEL
  if (allreduce) {
    MPI_Allreduce(&totag, &globally_reduced, 1, MPITypeMap<U>::type(), MPI_SUM,
                  MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&totag, &globally_reduced, 1, MPITypeMap<U>::type(), MPI_SUM, 0,
               MPI_COMM_WORLD);
  }
#else
  globally_reduced = totag;
#endif
}

KOKKOS_FORCEINLINE_FUNCTION bool fuzzy_equal(const Real &a, const Real &b, const Real &c,
                                             const Real &eps) {
  PARTHENON_DEBUG_REQUIRE(c > 0.0, "c input must be positive");
  // c input makes user decide metric
  return std::abs(a - b) < c * eps;
}

} // namespace jaybenne

#endif // JAYBENNE_JAYBENNE_UTILS_HPP_
