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

A portion of the CI is run on LANL's internal Darwin platform. To launch this CI job, someone with
Darwin access (usually a LANL employee) must first create a Github Personal Access Token, like so:

- `github.com` profile -> `Settings` -> `Developer Settings` -> `Personal Access Tokens` -> `Tokens (classic)`
- Click the `Generate New Token` button -> `Generate New Token (classic)`
- Name it something like `jaybenne_token` in the `Note` box
- Click the `workflow` checkbox (which will also check the `repo` boxes)
- `Generate token`
- You only get to see the token once, so immediately copy it.

Store the token securely in your own environment as `JAYBENNE_GITHUB_TOKEN`, e.g. in your Darwin `~/.bashrc`:

    export JAYBENNE_GITHUB_TOKEN=[token]

and then, again from Darwin, manually launch the CI runner:

    cd jaybenne
    ./tst/launch_ci_runner.py [Number of the github PR]

Note that `launch_ci_runner.py` will create a temporary checkout of the current state of the branch associated
with this PR according to the `origin` remote, so you don't need to worry about the state of your local checkout
of `jaybenne`.

# Run driver executable

    cd build/src
    mpiexec -n 1 ./mcblock -i ../../prob/jbinput.stepdiff
