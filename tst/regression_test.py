# ========================================================================================
#  (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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

import argparse
import os
import sys
import numpy as np
from subprocess import call
import shutil
import glob
import datetime

sys.path.insert(0, "../analysis/")
import jhdf
import __main__

# ------------------------------------------------------------------------------------------------ #
# Constants
#

ABS_TEST_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
now_hash = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ABS_BUILD_DIR = os.path.join(ABS_TEST_DIR, "build_" + now_hash)
ABS_RUN_DIR = os.path.join(ABS_TEST_DIR, "run_" + now_hash)
ABS_SOURCE_DIR = os.path.dirname(ABS_TEST_DIR)
TEMPORARY_INPUT_FILE = "test_input.pin"
SCRIPT_NAME = os.path.basename(__main__.__file__).split(".py")[0]

# ------------------------------------------------------------------------------------------------ #
# Utility functions
#


# -- Compare two values up to some floating point tolerance
def soft_equiv(val, ref, tol=1.0e-5):
    numerator = np.fabs(val - ref)
    denominator = max(np.fabs(ref), 1.0e-10)

    if numerator / denominator > tol:
        return False
    else:
        return True


# -- Read value of parameter in input file
def read_input_value(block, key, input_file):
    with open(input_file, "r") as infile:
        lines = infile.readlines()
        for line in lines:
            sline = line.strip()

            # Skip empty lines and comments
            if len(sline) == 0 or sline[0] == "#":
                continue

            # Check for block
            elif sline[0] == "<":
                current_block = sline.split("<")[1].split(">")[0]
                continue

            # Ignore multi-value lines
            elif len(sline.split("=")) != 2 or "," in sline or "&" in sline:
                continue

            else:
                current_key = sline.split("=")[0].strip()

                if block == current_block and key == current_key:
                    return sline.split("=")[1].strip()

    assert False, f'Input parameter "{block}/{key}" not found!'


# -- Modify key in input file, add key (and block) if not present, write new file
def modify_input(dict_key, value, input_file):
    key = dict_key.split("/")[-1]
    block = dict_key.split(key)[0][:-1]

    new_input_file = []

    current_block = None

    input_found = False

    with open(input_file, "r") as infile:
        lines = infile.readlines()
        for line in lines:
            sline = line.strip()

            # Skip empty lines and comments
            if len(sline) == 0 or sline[0] == "#":
                continue

            # Check for block
            elif sline[0] == "<":
                current_block = sline.split("<")[1].split(">")[0]

            # Check for key
            elif len(sline.split("=")) != 2 or "," in sline or "&" in sline:
                # Multiple values not supported for modification
                new_input_file.append(line)
                continue

            else:
                current_key = sline.split("=")[0].strip()
                current_value = sline.split("=")[1].strip()

                newline = line
                if block == current_block and key == current_key:
                    newline = key + " = " + str(value) + "\n"

                new_input_file.append(newline)
                input_found = True
                continue

            new_input_file.append(line)

    index = None
    if input_found == False:
        print(f'Input "{block}" "{key}" not found!')
        for i, line in enumerate(new_input_file):
            if line == f"<{block}>":
                index = i

    if index is None:
        # Block doesn't exist
        new_input_file.append(f"<{block}>\n")
        new_input_file.append(key + " = " + str(value) + "\n")
    else:
        # Block exists but key doesn't
        new_input_file.insert(index, key + " = " + str(value) + "\n")

    with open(input_file, "w") as outfile:
        for line in new_input_file:
            outfile.write(line)


# ------------------------------------------------------------------------------------------------ #
# Common regression test tools
#


# -- Default argument parser
def get_default_parser():
    parser = argparse.ArgumentParser(description="Regression testing")
    parser.add_argument(
        "--upgold",
        dest="upgold",
        action="store_true",
        help="Whether to overwrite the gold file rather than test against it.",
    )
    parser.add_argument(
        "--use_mpiexec",
        dest="use_mpiexec",
        action="store_true",
        help="Whether to launch the executable with mpiexec",
    )
    parser.add_argument(
        "--mpi_oversubscribe",
        dest="mpi_oversubscribe",
        action="store_true",
        help="Allow MPI to oversubscribe cores",
    )
    parser.add_argument(
        "--mpi_nthreads",
        type=int,
        default=1,
        help="Number of threads to use when running with MPI",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to input file",
    )
    parser.add_argument(
        "--executable",
        type=str,
        default=None,
        help="mcblock executable to use for testing",
    )
    parser.add_argument(
        "--build_type",
        type=str,
        default="Release",
        choices=["Debug", "Release"],
        help="Type of build to use",
    )
    parser.add_argument(
        "--cleanup",
        dest="cleanup",
        action="store_true",
        help="Whether to erase build/run directories created during testing.",
    )
    parser.add_argument(
        "--comparison",
        type=str,
        default="weighted_mean",
        help="Method of comparison against true solution at each point or via mean [pointwise, mean]",
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Whether to generate visual output.",
    )
    return parser


# -- Configure and build phoebus with problem-specific options
def build_code(geometry, build_type="Release"):
    if os.path.isdir(ABS_BUILD_DIR):
        print(
            f'ABS_BUILD_DIR "{ABS_BUILD_DIR}" already exists! Clean up before calling a regression test script!'
        )
        sys.exit(os.EX_SOFTWARE)

    import subprocess

    subprocess.run(["bash", "-c", f". ../env/bash; build_jaybenne -b {ABS_BUILD_DIR}"])


# -- Clean up working directory
def clean_up():
    def safer_rmtree(target_dir, parent_dir):
        target_path = os.path.normpath(target_dir)
        parent_path = os.path.normpath(parent_dir)

        if os.path.exists(target_path):
            if not parent_path == os.path.commonpath([target_path, parent_path]):
                print(
                    "Directory to remove is not inside repository directory! Don't trust this script to remove the directory!"
                )
                sys.exit(-1)

            protected_paths = ["/", "/etc", "/bin", "/usr"]
            if target_path in protected_paths:
                print(
                    f'You are trying to delete "{target_path}" which is protected! Don\'t trust this script to remove the directory!'
                )
                sys.exit(-1)

            print(f"Removing {target_path}")
            os.chdir("../")

            try:
                shutil.rmtree(target_path)
            except Exception as e:
                print(f"shutil.rmtree error: {e}")

    safer_rmtree(ABS_BUILD_DIR, ABS_SOURCE_DIR)
    safer_rmtree(ABS_RUN_DIR, ABS_SOURCE_DIR)


# -- Set up and run test problem
def run_problem(
    executable,
    build_type,
    input_file,
    modified_inputs,
    cleanup,
    use_mpiexec,
    oversubscribe,
    mpi_nthreads,
):
    if executable is None:
        executable = os.path.join(ABS_BUILD_DIR, "mcblock")
        build_code(geometry, build_type)
    else:
        if not os.path.isabs(executable):
            executable = os.path.abspath(executable)

    if cleanup:
        clean_up()

    if os.path.isdir(ABS_RUN_DIR):
        print(
            f'RUN_DIR "{ABS_RUN_DIR}" already exists! Clean up before calling a regression test script!'
        )
        sys.exit(os.EX_SOFTWARE)
    os.mkdir(ABS_RUN_DIR)
    os.chdir(ABS_RUN_DIR)

    # Copy test problem and modify inputs
    # shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../inputs/", input_file), TEMPORARY_INPUT_FILE)
    shutil.copyfile(input_file, TEMPORARY_INPUT_FILE)
    for key in modified_inputs:
        modify_input(key, modified_inputs[key], TEMPORARY_INPUT_FILE)

    # Run test problem
    preamble = []
    if use_mpiexec:
        preamble += ["mpiexec", "-n", f"{mpi_nthreads}"]
        if oversubscribe:
            preamble += ["--oversubscribe"]
    call(preamble + [executable, "-i", TEMPORARY_INPUT_FILE])

    # Get last dump file
    dumpfiles = np.sort(glob.glob("*.phdf"))
    if len(dumpfiles) == 0:
        print("Could not load any dump files!")
        sys.exit(os.EX_SOFTWARE)
    dump = jhdf.jhdf(dumpfiles[-1])

    return dump


# -- Run test problem with previously built code, input file, and modified inputs, and
#    compare to analytic expectation
def analytic_comparison(
    args,
    variables,
    solutions,
    modified_inputs={},
    tolerance=1.0e-10,
):

    input_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../inputs/", args.input
    )
    problem = read_input_value("parthenon/job", "problem_id", input_file)
    print("\n=== ANALYTIC COMPARISON TEST PROBLEM ===")
    print(f"= problem:      {problem}")
    print(f"= executable:   {args.executable}")
    print(f"= use_mpiexec:  {args.use_mpiexec}")
    if args.use_mpiexec:
        print(f"= oversubscribe: {args.mpi_oversubscribe}")
        print(f"= mpi_nthreads: {args.mpi_nthreads}")
    print(f"= build_type:   {args.build_type}")
    print(f"= tolerance:    {tolerance}")
    print(f"= cleanup:      {args.cleanup}")
    print(f"= comparison:   {args.comparison}")
    print(f"= visualize:    {args.visualize}")
    print("========================================\n")

    assert (
        args.comparison == "mean"
        or args.comparison == "pointwise"
        or args.comparison == "weighted_mean"
    ), 'Invalid "comparison" option!'

    dump = run_problem(
        args.executable,
        args.build_type,
        input_file,
        modified_inputs,
        args.cleanup,
        args.use_mpiexec,
        args.mpi_oversubscribe,
        args.mpi_nthreads,
    )

    # Loop over meshblocks and cells and compare each variable to its corresponding solution
    mean_error = 0
    max_error = 1.0e-100
    mean_frac_error = 0
    mean_frac_error_weighted = 0
    max_frac_error = 1.0e-100
    mean_count = 0
    weighted_norm = 0
    t = dump.Time
    success = True
    for nv, variable_name in enumerate(variables):
        print(variable_name)
        variable = dump.Get(variable_name)
        for nb in range(dump.NumBlocks):
            for k in range(dump.NX3):
                for j in range(dump.NX2):
                    for i in range(dump.NX1):
                        x = dump.X1c[nb, k, j, i]
                        y = dump.X2c[nb, k, j, i]
                        z = dump.X3c[nb, k, j, i]
                        error = np.fabs(
                            solutions[nv](t, x, y, z) - variable[nb, k, j, i]
                        )
                        mean_error += error
                        if error > max_error:
                            max_error = error
                        frac_error = error / np.fabs(
                            (solutions[nv](t, x, y, z) + variable[nb, k, j, i]) / 2
                        )
                        mean_frac_error += frac_error
                        if frac_error > max_frac_error:
                            max_frac_error = frac_error
                        mean_count += 1
                        mean_frac_error_weighted += frac_error * solutions[nv](
                            t, x, y, z
                        )
                        weighted_norm += solutions[nv](t, x, y, z)
                        if args.comparison == "pointwise":
                            if frac_error > tolerance:
                                success = False

    mean_error /= mean_count
    mean_frac_error /= mean_count
    mean_frac_error_weighted /= weighted_norm

    print(f"Mean error:                     {mean_error:.2e}")
    print(f"Mean fractional error:          {mean_frac_error:.2e}")
    print(f"Mean weighted fractional error: {mean_frac_error_weighted:.2e}")
    print(f"Max error:                      {max_error:.2e}")
    print(f"Max fractional error:           {max_frac_error:.2e}")

    if args.comparison == "mean":
        if mean_frac_error > tolerance:
            success = False
    elif args.comparison == "weighted_mean":
        if mean_frac_error_weighted > tolerance:
            success = False

    if args.visualize:
        import matplotlib.pyplot as plt

        for nv, variable_name in enumerate(variables):
            variable = dump.Get(variable_name)
            fig, ax = plt.subplots(1, 1)
            for nb in range(dump.NumBlocks):
                x = dump.X1c[nb, 0, 0, :]
                y_sol = np.zeros_like(x)
                for i in range(dump.NX1):
                    y_sol = solutions[nv](t, x, 0, 0)
                ax.plot(x, y_sol)
                ax.plot(x, variable[nb, 0, 0, :])
            plt.savefig(f"../analytic_compare_{variable_name}.png")
            plt.clf()

    if args.cleanup == True:
        clean_up()

    if success:
        print("TEST PASSED")
        return os.EX_OK
    else:
        print("TEST FAILED")
        return os.EX_SOFTWARE


# -- Run test problem with previously built code, input file, and modified inputs, and
#    compare to gold output
def gold_comparison(
    args,
    variables,
    modified_inputs={},
    compression_factor=1,
    tolerance=0.2,
):
    input_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../inputs/", args.input
    )
    problem = read_input_value("parthenon/job", "problem_id", input_file)
    print("\n=== GOLD COMPARISON TEST PROBLEM ===")
    print(f"= problem:       {problem}")
    print(f"= executable:    {args.executable}")
    print(f"= use_mpiexec:   {args.use_mpiexec}")
    if args.use_mpiexec:
        print(f"= oversubscribe: {args.mpi_oversubscribe}")
    print(f"= build_type:    {args.build_type}")
    print(f"= compression:   {compression_factor}")
    print(f"= tolerance:     {tolerance}")
    print(f"= cleanup:       {args.cleanup}")
    print(f"= comparison:    {args.comparison}")
    print("====================================\n")

    assert (
        args.comparison == "mean" or args.comparison == "pointwise"
    ), 'Invalid "comparison" option!'

    dump = run_problem(
        args.executable,
        args.build_type,
        input_file,
        modified_inputs,
        args.cleanup,
        args.use_mpiexec,
        args.mpi_oversubscribe,
    )

    # Construct array of results values
    variables_data = np.empty(shape=(0))
    for variable_name in variables:
        variable = dump.Get(variable_name)
        if len(variable.shape) > 1:
            dim = variable.shape[0]
            for d in range(dim):
                variables_data = np.concatenate((variables_data, variable[d, :]))
        else:
            variables_data = np.concatenate((variables_data, variable))

    # Compress results, if desired
    compression_factor = int(compression_factor)
    compressed_variables = np.zeros(len(variables_data) // compression_factor)
    for n in range(len(compressed_variables)):
        compressed_variables[n] = variables_data[compression_factor * n]
    variables_data = compressed_variables

    # Write gold file, or compare to existing gold file
    success = True
    gold_name = os.path.join("../", SCRIPT_NAME) + ".gold"
    if args.upgold:
        np.savetxt(gold_name, variables_data, newline="\n")
    else:
        gold_variables = np.loadtxt(gold_name)
        if not len(gold_variables) == len(variables_data):
            print("Length of gold variables does not match calculated variables!")
            success = False
        else:
            for n in range(len(gold_variables)):
                if not soft_equiv(variables_data[n], gold_variables[n], tol=tolerance):
                    if args.comparison == "pointwise":
                        success = False

        norm = np.clip((variables_data + gold_variables) / 2, 1.0e-100, None)
        mean_error = np.mean(np.fabs(variables_data - gold_variables))
        mean_frac_error = np.mean(np.fabs(variables_data - gold_variables) / norm)
        max_error = np.max(np.fabs(variables_data - gold_variables))
        max_frac_error = np.max(np.fabs(variables_data - gold_variables) / norm)

        if args.comparison == "mean":
            if mean_frac_error > tolerance:
                success = False

    if args.cleanup == True:
        clean_up()

    # Report upgolding, success, or failure
    if args.upgold:
        print(f"Gold file {gold_name} updated!")
    else:
        print(f"Mean error:            {mean_error:.2e}")
        print(f"Mean fractional error: {mean_frac_error:.2e}")
        print(f"Max error:             {max_error:.2e}")
        print(f"Max fractional error:  {max_frac_error:.2e}")
        if success:
            print("TEST PASSED")
            return os.EX_OK
        else:
            print("TEST FAILED")
            return os.EX_SOFTWARE
