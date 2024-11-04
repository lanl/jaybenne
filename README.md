# Jaybenne

Jaybenne IMC code for photon transport in the performance-portable block-structured AMR
framework parthenon (https://github.com/lanl/parthenon)

Contributors: Ben R. Ryan, Patrick Mullen, Alex Long, Ryan Wollaeger

For software licensing refer to LICENSE.md

Copyright assertion O4812

# Required dependencies

* CMake 3.13 or greater
* C++ compiler (C++17 support)
* MPI
* OpenMP
* HDF5
* Parthenon

# Environment

On deployed platforms, the environment with required dependencies can be set up via

    source env/bash

Currently supported computers/partitions are:

## Darwin

    power9-rhel7
    volta-x86
    skylake-gold

# Submodules

Dependencies (parthenon, singularity-*) are ingested via submodules and compiled alongside
jaybenne. The most reliable way to update submodule versions is to simply remove the
`external` folder:

    rm -rf external/

and re-initialize the submodules

    git submodule update --init --recursive

# Installation

    git submodule update --init --recursive
    mkdir build
    cd build
    cmake ../
    make -j

# Formatting the software

Any contributions to the ngPFC software must be compliant with the C++ clang formatter and
the Python black formatter.  Linting contributions can be done via an automated formatting
script:
`CFM=clang-format-12 ./style/format.sh`

# CI

We use the gitlab CI for regression testing. The CI will not run if the PR is marked "Draft:" or
"WIP:". Removing these labels from the title will not automatically launch the CI. To launch the CI
with an empty commit, do

    git commit --allow-empty -m "trigger pipeline" && git push

# Run driver executable

    cd build/src
    mpiexec -n 1 ./mcblock -i ../../prob/jbinput.stepdiff
