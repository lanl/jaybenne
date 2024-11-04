#!/bin/bash
#-----------------------------------------------------------------------------------------
# (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S.  Department of Energy/National
# Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of
# Energy/National Nuclear Security Administration. The Government is
# granted for itself and others acting on its behalf a nonexclusive,
# paid-up, irrevocable worldwide license in this material to reproduce,
# prepare derivative works, distribute copies to the public, perform
# publicly and display publicly, and to permit others to do so.
#-----------------------------------------------------------------------------------------

# NOTE(@pdmullen): the following is largely borrowed from the open-source LANL phoebus
# software

: ${CFM:=clang-format}
: ${PFM:=black}
: ${VERBOSE:=0}

if ! command -v ${CFM} &> /dev/null; then
    >&2 echo "Error: No clang format found! Looked for ${CFM}"
    return 1
else
    CFM=$(command -v ${CFM})
    echo "Clang format found: ${CFM}"
fi

# clang format major version
TARGET_CF_VRSN=12
CF_VRSN=$(${CFM} --version)
echo "Note we assume clang format version ${TARGET_CF_VRSN}."
echo "You are using ${CF_VRSN}."
echo "If these differ, results may not be stable."

echo "Formatting C++ files..."
REPO=$(git rev-parse --show-toplevel)
for f in $(git ls-tree --full-tree --name-only -r HEAD | grep -E 'cpp$|hpp$'); do
    if [ ${VERBOSE} -ge 1 ]; then
       echo ${f}
    fi
    ${CFM} -i ${REPO}/${f}
done
echo "...Done"

# format python files
if ! command -v ${PFM} &> /dev/null; then
    >&2 echo "Error: No version of black found! Looked for ${PFM}"
    return 1
else
    PFM=$(command -v ${PFM})
    echo "black Python formatter found: ${PFM}"
    echo "black version: $(${PFM} --version)"
fi

echo "Formatting Python files..."
REPO=$(git rev-parse --show-toplevel)
for f in $(git ls-tree --full-tree --name-only -r HEAD | grep -E 'py'); do
    if [ ${VERBOSE} -ge 1 ]; then
       echo ${f}
    fi
    ${PFM} -q ${REPO}/${f}
done
echo "...Done"
