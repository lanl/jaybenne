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

// Parthenon includes
#include <utils/robust.hpp>

// Jaybenne includes
#include "jaybenne.hpp"
#include "jaybenne_utils.hpp"
#include "transport_utils.hpp"

namespace jaybenne {

KOKKOS_FORCEINLINE_FUNCTION
void SampleFace2D(const int i_l, const Real dx, const Real P_l, const Real P_u,
                  RngGen &rng_gen, int &i, Real &x) {

  const int i_u = i_l + 1;

  // sample face
  const Real xi = (P_l + P_u) * rng_gen.drand();
  if (xi < P_l) {
    // sample lower face
    x -= dx * rng_gen.drand();
    i = i_l;
  } else {
    // sample upper face
    x += dx * rng_gen.drand();
    i = i_u;
  }
}

KOKKOS_FORCEINLINE_FUNCTION
void SampleFace3D(const int i1_l, const int i2_l, const Real dx1, const Real dx2,
                  const Real P_ll, const Real P_lu, const Real P_ul, const Real P_uu,
                  RngGen &rng_gen, int &i1, int &i2, Real &x1, Real &x2) {

  const int i1_u = i1_l + 1;
  const int i2_u = i2_l + 1;

  // sample face
  const Real xi = (P_ll + P_lu + P_ul + P_uu) * rng_gen.drand();
  if (xi < P_ll) {
    // sample (low-x1,low-x2) face
    x1 -= dx1 * rng_gen.drand();
    i1 = i1_l;
    x2 -= dx2 * rng_gen.drand();
    i2 = i2_l;
  } else if (xi < P_ll + P_lu) {
    // sample (upper-x1,low-x2) face
    x1 += dx1 * rng_gen.drand();
    i1 = i1_u;
    x2 -= dx2 * rng_gen.drand();
    i2 = i2_l;
  } else if (xi < P_ll + P_lu + P_ul) {
    // sample (low-x1, upper-x2) face
    x1 -= dx1 * rng_gen.drand();
    i1 = i1_l;
    x2 += dx2 * rng_gen.drand();
    i2 = i2_u;
  } else {
    // sample (upper-x1, upper-x2) face
    x1 += dx1 * rng_gen.drand();
    i1 = i1_u;
    x2 += dx2 * rng_gen.drand();
    i2 = i2_u;
  }
}

// sample face for particles coming from a coarser block DDMC cell
TaskStatus SampleDDMCBlockFace(MeshData<Real> *md) {
  namespace fj = field::jaybenne;
  namespace fjh = field::jaybenne::host;
  namespace sp = swarm_position;
  namespace ph = particle::photons;
  using TE = parthenon::TopologicalElement;

  auto pm = md->GetParentPointer();
  // do nothing for 1D
  if (!(pm->ndim > 1)) return TaskStatus::complete;

  auto &resolved_pkgs = pm->resolved_packages;
  auto &jb_pkg = pm->packages.Get("jaybenne");
  auto &rng_pool = jb_pkg->Param<RngPool>("rng_pool");
  const Real vv = jb_pkg->Param<Real>("speed_of_light");

  // get dimension indicators (for face probabilities)
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  // Create SparsePack
  static auto desc = MakePackDescriptor<fj::ddmc_face_prob>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // set tolerance for checking particle coordinate
  constexpr Real eps = parthenon::robust::EPS();

  // Create SwarmPacks
  static auto pdesc_r =
      MakeSwarmPackDescriptor<sp::x, sp::y, sp::z, ph::v>(photons_swarm_name);
  static auto pdesc_i = MakeSwarmPackDescriptor<ph::ijk>(photons_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  // Indexing and dimensionality
  const int &nparticles_per_pack = ppack_r.GetMaxFlatIndex();

  // identify DDMC criteria from previous block
  if (!three_d) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "SampleDDMCBlockFace::2D", DevExecSpace(), 0,
        nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
          auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
          const auto &swarm_d = ppack_r.GetContext(b);
          if (swarm_d.IsActive(n)) {

            Real &vx = ppack_r(b, ph::v(0), n);
            Real &vy = ppack_r(b, ph::v(1), n);
            Real &vz = ppack_r(b, ph::v(2), n);

            // speed, direction were set to 0 if particle was DDMC and moved off block
            if (vx * vx + vy * vy + vz * vz < eps * vv * vv) {

              // get particle coordinate, and stale cell index
              Real &x = ppack_r(b, sp::x(), n);
              Real &y = ppack_r(b, sp::y(), n);
              Real &z = ppack_r(b, sp::z(), n);
              int &ip = ppack_i(b, ph::ijk(0), n);
              int &jp = ppack_i(b, ph::ijk(1), n);
              int &kp = ppack_i(b, ph::ijk(2), n);

              // get cell of particle
              swarm_d.Xtoijk(x, y, z, ip, jp, kp);

              auto &coords = vmesh.GetCoordinates(b);
              const Real &dx_i = coords.Dxc<parthenon::X1DIR>(0, 0, 0);
              const Real &dy_j = coords.Dxc<parthenon::X2DIR>(0, 0, 0);
              // low coordinates of particle's current cell
              const Real x_i = coords.Xc<parthenon::X1DIR>(ip) - 0.5 * dx_i;
              const Real y_j = coords.Xc<parthenon::X2DIR>(jp) - 0.5 * dy_j;

              // get rng state for face and direction sampling
              auto rng_gen = rng_pool.get_state();

              // check particle proximity to block faces
              // if DDMC moves particle eps_ddmc_offset coarse cell dx then
              // 2*eps_ddmc_offset current block cell dx
              const bool at_block_x_min = fuzzy_equal(
                  x, swarm_d.x_min_ + 2.0 * eps_ddmc_offset * dx_i, dx_i, eps);
              const bool at_block_x_max = fuzzy_equal(
                  x, swarm_d.x_max_ - 2.0 * eps_ddmc_offset * dx_i, dx_i, eps);
              const bool at_block_y_min = fuzzy_equal(
                  y, swarm_d.y_min_ + 2.0 * eps_ddmc_offset * dy_j, dy_j, eps);
              const bool at_block_y_max = fuzzy_equal(
                  y, swarm_d.y_max_ - 2.0 * eps_ddmc_offset * dy_j, dy_j, eps);

              if (at_block_x_min || at_block_x_max) {
                // check at high or low x block face

                // sample velocity
                const Real dir_sgn = at_block_x_min ? 1.0 : -1.0;
                sample_face_iso_dir(dir_sgn * vv, rng_gen, vx, vy, vz);

                // select X1 face index for accessing DDMC probabilities
                const int ip_b = at_block_x_min ? ip : ip + 1;

                // check if at y-bound between two faces along the y-dirction
                const bool at_y_edge_u = fuzzy_equal(y, y_j, dy_j, eps);
                const bool at_y_edge_l = fuzzy_equal(y, y_j + dy_j, dy_j, eps);

                // check if particle is at a y-edge of its current cell
                if (at_y_edge_u || at_y_edge_l) {

                  const int jp_u = at_y_edge_u ? jp : jp + 1;
                  const int jp_l = at_y_edge_u ? jp - 1 : jp;

                  // get face probabilities for bounding faces
                  const Real &Px_uy =
                      vmesh(b, TE::F1, fj::ddmc_face_prob(), kp, jp_u, ip_b);
                  const Real &Px_ly =
                      vmesh(b, TE::F1, fj::ddmc_face_prob(), kp, jp_l, ip_b);

                  SampleFace2D(jp_l, dy_j, Px_ly, Px_uy, rng_gen, jp, y);
                }

              } else if (at_block_y_min || at_block_y_max) {
                // at high or low y block face

                // sample velocity
                const Real dir_sgn = at_block_y_min ? 1.0 : -1.0;
                sample_face_iso_dir(dir_sgn * vv, rng_gen, vy, vz, vx);

                // select X2 face index for accessing DDMC probabilities
                const int jp_b = at_block_y_min ? jp : jp + 1;

                // check if at x-bound between two faces along the x-dirction
                const bool at_x_edge_u = fuzzy_equal(x, x_i, dx_i, eps);
                const bool at_x_edge_l = fuzzy_equal(x, x_i + dx_i, dx_i, eps);

                // check if particle is at a x-edge of its current cell
                if (at_x_edge_u || at_x_edge_l) {

                  const int ip_u = at_x_edge_u ? ip : ip + 1;
                  const int ip_l = at_x_edge_u ? ip - 1 : ip;

                  // get face probabilities for bounding faces
                  const Real &Py_ux =
                      vmesh(b, TE::F2, fj::ddmc_face_prob(), kp, jp_b, ip_u);
                  const Real &Py_lx =
                      vmesh(b, TE::F2, fj::ddmc_face_prob(), kp, jp_b, ip_l);

                  SampleFace2D(ip_l, dx_i, Py_lx, Py_ux, rng_gen, ip, x);
                }
              }

              rng_pool.free_state(rng_gen);

              // Check particle is still on block
              PARTHENON_DEBUG_REQUIRE(x >= swarm_d.x_min_ || x <= swarm_d.x_max_,
                                      "Particle sampled outside of meshblock!");
              PARTHENON_DEBUG_REQUIRE(y >= swarm_d.y_min_ || y <= swarm_d.y_max_,
                                      "Particle sampled outside of meshblock!");
              PARTHENON_DEBUG_REQUIRE(z >= swarm_d.z_min_ || z <= swarm_d.z_max_,
                                      "Particle sampled outside of meshblock!");
            }
          }
        });
  } else {

    // Do 3D version
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "SampleDDMCBlockFace::3D", DevExecSpace(), 0,
        nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
          auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
          const auto &swarm_d = ppack_r.GetContext(b);
          if (swarm_d.IsActive(n)) {

            Real &vx = ppack_r(b, ph::v(0), n);
            Real &vy = ppack_r(b, ph::v(1), n);
            Real &vz = ppack_r(b, ph::v(2), n);

            // speed, direction were set to 0 if particle was DDMC and moved off block
            if (vx * vx + vy * vy + vz * vz < eps * vv * vv) {

              // get particle coordinate, and (stale?) cell index
              Real &x = ppack_r(b, sp::x(), n);
              Real &y = ppack_r(b, sp::y(), n);
              Real &z = ppack_r(b, sp::z(), n);
              int &ip = ppack_i(b, ph::ijk(0), n);
              int &jp = ppack_i(b, ph::ijk(1), n);
              int &kp = ppack_i(b, ph::ijk(2), n);

              // get cell of particle
              swarm_d.Xtoijk(x, y, z, ip, jp, kp);

              auto &coords = vmesh.GetCoordinates(b);
              const Real &dx_i = coords.Dxc<parthenon::X1DIR>(0, 0, 0);
              const Real &dy_j = coords.Dxc<parthenon::X2DIR>(0, 0, 0);
              const Real &dz_k = coords.Dxc<parthenon::X3DIR>(0, 0, 0);
              // low coordinates of particle's current cell
              const Real x_i = coords.Xc<parthenon::X1DIR>(ip) - 0.5 * dx_i;
              const Real y_j = coords.Xc<parthenon::X2DIR>(jp) - 0.5 * dy_j;
              const Real z_k = coords.Xc<parthenon::X3DIR>(kp) - 0.5 * dz_k;

              // get rng state for face and direction sampling
              auto rng_gen = rng_pool.get_state();

              // check particle proximity to block faces
              // if DDMC moves particle eps_ddmc_offset coarse cell dx then
              // 2*eps_ddmc_offset current block cell dx
              const bool at_block_x_min = fuzzy_equal(
                  x, swarm_d.x_min_ + 2.0 * eps_ddmc_offset * dx_i, dx_i, eps);
              const bool at_block_x_max = fuzzy_equal(
                  x, swarm_d.x_max_ - 2.0 * eps_ddmc_offset * dx_i, dx_i, eps);
              const bool at_block_y_min = fuzzy_equal(
                  y, swarm_d.y_min_ + 2.0 * eps_ddmc_offset * dy_j, dy_j, eps);
              const bool at_block_y_max = fuzzy_equal(
                  y, swarm_d.y_max_ - 2.0 * eps_ddmc_offset * dy_j, dy_j, eps);
              const bool at_block_z_min = fuzzy_equal(
                  z, swarm_d.z_min_ + 2.0 * eps_ddmc_offset * dz_k, dz_k, eps);
              const bool at_block_z_max = fuzzy_equal(
                  z, swarm_d.z_max_ - 2.0 * eps_ddmc_offset * dz_k, dz_k, eps);

              if (at_block_x_min || at_block_x_max) {
                // check at high or low x block face

                // sample velocity
                const Real dir_sgn = at_block_x_min ? 1.0 : -1.0;
                sample_face_iso_dir(dir_sgn * vv, rng_gen, vx, vy, vz);

                // select X1 face index for accessing DDMC probabilities
                const int ip_b = at_block_x_min ? ip : ip + 1;

                // check if at yz-center of the four faces
                const bool at_y_edge_u = fuzzy_equal(y, y_j, dy_j, eps);
                const bool at_y_edge_l = fuzzy_equal(y, y_j + dy_j, dy_j, eps);
                const bool at_z_edge_u = fuzzy_equal(z, z_k, dz_k, eps);
                const bool at_z_edge_l = fuzzy_equal(z, z_k + dz_k, dz_k, eps);

                // check if particle is at a vertex of its current cell
                if ((at_y_edge_u || at_y_edge_l) && (at_z_edge_u || at_z_edge_l)) {

                  const int jp_u = at_y_edge_u ? jp : jp + 1;
                  const int jp_l = at_y_edge_u ? jp - 1 : jp;
                  const int kp_u = at_z_edge_u ? kp : kp + 1;
                  const int kp_l = at_z_edge_u ? kp - 1 : kp;

                  // get face probabilities for bounding faces
                  const Real &Px_uu =
                      vmesh(b, TE::F1, fj::ddmc_face_prob(), kp_u, jp_u, ip_b);
                  const Real &Px_ul =
                      vmesh(b, TE::F1, fj::ddmc_face_prob(), kp_u, jp_l, ip_b);
                  const Real &Px_lu =
                      vmesh(b, TE::F1, fj::ddmc_face_prob(), kp_l, jp_u, ip_b);
                  const Real &Px_ll =
                      vmesh(b, TE::F1, fj::ddmc_face_prob(), kp_l, jp_l, ip_b);

                  // sample a refined yz-face
                  SampleFace3D(jp_l, kp_l, dy_j, dz_k, Px_ll, Px_lu, Px_ul, Px_uu,
                               rng_gen, jp, kp, y, z);
                }

              } else if (at_block_y_min || at_block_y_max) {
                // at high or low y block face

                // sample velocity
                const Real dir_sgn = at_block_y_min ? 1.0 : -1.0;
                sample_face_iso_dir(dir_sgn * vv, rng_gen, vy, vz, vx);

                // select X2 face index for accessing DDMC probabilities
                const int jp_b = at_block_y_min ? jp : jp + 1;

                // check if at zx-center of the four faces
                const bool at_z_edge_u = fuzzy_equal(z, z_k, dz_k, eps);
                const bool at_z_edge_l = fuzzy_equal(z, z_k + dz_k, dz_k, eps);
                const bool at_x_edge_u = fuzzy_equal(x, x_i, dx_i, eps);
                const bool at_x_edge_l = fuzzy_equal(x, x_i + dx_i, dx_i, eps);

                // check if particle is at a vertex of its current cell
                if ((at_z_edge_u || at_z_edge_l) && (at_x_edge_u || at_x_edge_l)) {

                  const int kp_u = at_z_edge_u ? kp : kp + 1;
                  const int kp_l = at_z_edge_u ? kp - 1 : kp;
                  const int ip_u = at_x_edge_u ? ip : ip + 1;
                  const int ip_l = at_x_edge_u ? ip - 1 : ip;

                  // get face probabilities for bounding faces
                  const Real &Py_uu =
                      vmesh(b, TE::F2, fj::ddmc_face_prob(), kp_u, jp_b, ip_u);
                  const Real &Py_ul =
                      vmesh(b, TE::F2, fj::ddmc_face_prob(), kp_u, jp_b, ip_l);
                  const Real &Py_lu =
                      vmesh(b, TE::F2, fj::ddmc_face_prob(), kp_l, jp_b, ip_u);
                  const Real &Py_ll =
                      vmesh(b, TE::F2, fj::ddmc_face_prob(), kp_l, jp_b, ip_l);

                  // sample a refined zx-face
                  SampleFace3D(ip_l, kp_l, dx_i, dz_k, Py_ll, Py_lu, Py_ul, Py_uu,
                               rng_gen, ip, kp, x, z);
                }

              } else if (at_block_z_min || at_block_z_max) {
                // at high or low z block face

                // sample velocity
                const Real dir_sgn = at_block_z_min ? 1.0 : -1.0;
                sample_face_iso_dir(dir_sgn * vv, rng_gen, vz, vx, vy);

                // select X3 face index for accessing DDMC probabilities
                const int kp_b = at_block_z_min ? kp : kp + 1;

                // check if at xy-center of the four faces
                const bool at_x_edge_u = fuzzy_equal(x, x_i, dx_i, eps);
                const bool at_x_edge_l = fuzzy_equal(x, x_i + dx_i, dx_i, eps);
                const bool at_y_edge_u = fuzzy_equal(y, y_j, dy_j, eps);
                const bool at_y_edge_l = fuzzy_equal(y, y_j + dy_j, dy_j, eps);

                // check if particle is at a vertex of its current cell
                if ((at_x_edge_u || at_x_edge_l) && (at_y_edge_u || at_y_edge_l)) {

                  const int ip_u = at_x_edge_u ? ip : ip + 1;
                  const int ip_l = at_x_edge_u ? ip - 1 : ip;
                  const int jp_u = at_y_edge_u ? jp : jp + 1;
                  const int jp_l = at_y_edge_u ? jp - 1 : jp;

                  // get face probabilities for bounding faces
                  const Real &Pz_uu =
                      vmesh(b, TE::F3, fj::ddmc_face_prob(), kp_b, jp_u, ip_u);
                  const Real &Pz_ul =
                      vmesh(b, TE::F3, fj::ddmc_face_prob(), kp_b, jp_u, ip_l);
                  const Real &Pz_lu =
                      vmesh(b, TE::F3, fj::ddmc_face_prob(), kp_b, jp_l, ip_u);
                  const Real &Pz_ll =
                      vmesh(b, TE::F3, fj::ddmc_face_prob(), kp_b, jp_l, ip_l);

                  // sample a refined xy-face
                  SampleFace3D(ip_l, jp_l, dx_i, dy_j, Pz_ll, Pz_lu, Pz_ul, Pz_uu,
                               rng_gen, ip, jp, x, y);
                }
              }

              rng_pool.free_state(rng_gen);

              // Check particle is still on block
              PARTHENON_DEBUG_REQUIRE(x >= swarm_d.x_min_ || x <= swarm_d.x_max_,
                                      "Particle sampled outside of meshblock!");
              PARTHENON_DEBUG_REQUIRE(y >= swarm_d.y_min_ || y <= swarm_d.y_max_,
                                      "Particle sampled outside of meshblock!");
              PARTHENON_DEBUG_REQUIRE(z >= swarm_d.z_min_ || z <= swarm_d.z_max_,
                                      "Particle sampled outside of meshblock!");
            }
          }
        });
  }

  return TaskStatus::complete;
}

} // namespace jaybenne
