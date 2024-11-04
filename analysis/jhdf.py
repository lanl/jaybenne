# ========================================================================================
#  (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

import sys
import os
import numpy as np

# Provide path directly to phdf to avoid need to install parthenon_tools package
jhdf_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.insert(
    0,
    os.path.join(
        jhdf_dir,
        "../external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools/",
    ),
)
import phdf


# Class for opening a Jaybenne dump file and providing access to data.
# Wraps the Parthenon phdf class and provides additional Jaybenne-specific data.
class jhdf(phdf.phdf):
    def __init__(self, filename):
        super().__init__(filename)

        # Get meshblock coordinate properties
        self.NX1 = self.MeshBlockSize[0]
        self.NX2 = self.MeshBlockSize[1]
        self.NX3 = self.MeshBlockSize[2]
        self.DX1 = np.zeros(self.NumBlocks)
        self.DX2 = np.zeros(self.NumBlocks)
        self.DX3 = np.zeros(self.NumBlocks)
        for b in range(self.NumBlocks):
            self.DX1[b] = (self.BlockBounds[b][1] - self.BlockBounds[b][0]) / self.NX1
            self.DX2[b] = (self.BlockBounds[b][3] - self.BlockBounds[b][2]) / self.NX2
            self.DX3[b] = (self.BlockBounds[b][5] - self.BlockBounds[b][4]) / self.NX3

        # Get node coordinates for each meshblock for plotting
        self.X1n = np.zeros([self.NumBlocks, self.NX3 + 1, self.NX2 + 1, self.NX1 + 1])
        self.X2n = np.zeros([self.NumBlocks, self.NX3 + 1, self.NX2 + 1, self.NX1 + 1])
        self.X3n = np.zeros([self.NumBlocks, self.NX3 + 1, self.NX2 + 1, self.NX1 + 1])
        self.X1c = np.zeros([self.NumBlocks, self.NX3, self.NX2, self.NX1])
        self.X2c = np.zeros([self.NumBlocks, self.NX3, self.NX2, self.NX1])
        self.X3c = np.zeros([self.NumBlocks, self.NX3, self.NX2, self.NX1])
        for b in range(self.NumBlocks):
            for k in range(self.NX3 + 1):
                for j in range(self.NX2 + 1):
                    for i in range(self.NX1 + 1):
                        self.X1n[b, k, j, i] = self.BlockBounds[b][0] + i * self.DX1[b]
                        self.X2n[b, k, j, i] = self.BlockBounds[b][2] + j * self.DX2[b]
                        self.X3n[b, k, j, i] = self.BlockBounds[b][4] + k * self.DX3[b]
            for k in range(self.NX3):
                for j in range(self.NX2):
                    for i in range(self.NX1):
                        self.X1c[b, k, j, i] = (
                            self.BlockBounds[b][0] + (i + 0.5) * self.DX1[b]
                        )
                        self.X2c[b, k, j, i] = (
                            self.BlockBounds[b][2] + (j + 0.5) * self.DX2[b]
                        )
                        self.X3c[b, k, j, i] = (
                            self.BlockBounds[b][4] + (k + 0.5) * self.DX3[b]
                        )

        self.xn = self.X1n
        self.yn = self.X2n
        self.zn = self.X3n
        self.xc = self.X1c
        self.yc = self.X2c
        self.zc = self.X3c

    # Returns data for a particular variable. Reports variables available in the dump
    # file if an invalid variable_name is provided.
    def Get(self, variable_name, flatten=False, report_available=True):
        variable = super().Get(variable_name, flatten)

        if variable is None and report_available:
            print("Variables contained in this dump file:")
            for name in self.Variables:
                if name not in [
                    "Blocks",
                    "Info",
                    "Input",
                    "Levels",
                    "Locations",
                    "LogicalLocations",
                    "Params",
                    "SparseInfo",
                    "VolumeLocations",
                ]:
                    print(f"  {name}")
            print("")

        return variable
