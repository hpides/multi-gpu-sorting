import argparse
import pathlib
import subprocess
import time

import settings

if __name__ == "__main__":
    settings.init()

    parser = argparse.ArgumentParser(description="run multi-GPU experiments",
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=60))
    parser.add_argument("-e",
                        "--experiments",
                        metavar="EXPERIMENTS",
                        help="the experiments",
                        action="store",
                        type=str,
                        default="",
                        dest="experiments")
    arguments = parser.parse_args()

    experiments = list(filter(None, arguments.experiments.split(",")))

    script_path = pathlib.Path(__file__).parent.resolve()
    executable_path = pathlib.Path(script_path / settings.executables_path).resolve()
    output_path = pathlib.Path(script_path / settings.experiments_path / time.strftime("%Y_%m_%d_%H_%M_%S")).resolve()

    output_path.mkdir(parents=True, exist_ok=True)

    for experiment in settings.experiments:
        if experiments and experiment.identifier not in experiments:
            continue

        for index, arguments in enumerate(experiment.arguments):
            numactl = "numactl -m 0" if ("gnu_parallel" in arguments or "paradis" in arguments) else "numactl -N 0 -m 0"

            command = "%s %s" % (numactl, pathlib.Path(executable_path / experiment.executable).resolve())
            for (parameter, argument) in zip(experiment.parameters, arguments):
                command += " %s %s" % (parameter, argument)

            for repetition in range(experiment.repetitions):
                columns = "%s\n" % (",".join(
                    "\"%s\"" % (column) for column in experiment.columns)) if index == 0 and repetition == 0 else ""

                if experiment.profilers:
                    for profiler in experiment.profilers:
                        if profiler == "nvprof" or profiler == "nsys":
                            output_file = pathlib.Path(
                                output_path /
                                ("%s_%s_%s_%s" %
                                 (experiment.executable, experiment.identifier, index + 1, repetition + 1))).resolve()

                            if profiler == "nvprof":
                                command = ";".join([
                                    "nvprof --csv --log-file %s.csv %s" % (output_file, command),
                                    "nvprof --quiet --export-profile %s.nvvp %s" % (output_file, command)
                                ])
                            elif profiler == "nsys":
                                command = "nsys profile -o %s.qdrep %s" % (output_file, command)

                            output = subprocess.run(command,
                                                    stdout=subprocess.PIPE,
                                                    universal_newlines=True,
                                                    shell=True)

                            if output.returncode == 0:
                                print("%s%s" % (columns, output.stdout), end="")

                else:
                    output = subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True, shell=True)

                    if output.returncode == 0:
                        print("%s%s" % (columns, output.stdout), end="")

                        output_file = pathlib.Path(
                            output_path / ("%s_%s.csv" % (experiment.executable, experiment.identifier))).resolve()

                        with output_file.open("a") as f:
                            f.write("%s%s" % (columns, output.stdout))
