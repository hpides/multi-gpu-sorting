import argparse
import matplotlib as plotlib
import matplotlib.pyplot as pyplot
import matplotlib.container as pycontainer
import numpy
import pandas
import pathlib

import settings

from typing import Dict, List, Union

plotlib.rcParams['hatch.linewidth'] = 0.5
plotlib.rcParams["mathtext.fontset"] = "cm"
plotlib.rc("font", **{"family": "Linux Libertine O"})

figure_width = 4.85
figure_height = 3.85


def scale_figure_size(width_factor: float, height_factor: float):
    pyplot.figure(num=1, figsize=(figure_width * width_factor, figure_height * height_factor))


legend_font_size = 17
small_font_size = 21
large_font_size = 23

colors = ["#3193C6", "#05AD97", "#AAC56C", "#F7AB13", "#CD4E38"]
hatches = ["//////", "\\\\\\\\\\\\", "xxxxxx", "......", "oooooo"]


def calculate_means(rows: List[str]):
    return [numpy.array([[float(value) for value in row.split(",")] for row in rows]).mean(axis=0)]


def annotate_bars(bars: pycontainer.BarContainer, precision: int, height: float = None, rotation: str = None):
    bar = bars[0]
    bar_height = height or bar.get_height()

    value = str(int(bar_height)) if precision == 0 else ("{:.%sf}" % (precision)).format(bar_height)

    pyplot.annotate(value,
                    xy=(bar.get_x() + bar.get_width() / 2, bar_height),
                    xytext=(0, 5 if rotation is not None else 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=rotation,
                    fontsize=small_font_size,
                    zorder=3)


def plot_bars(data: Dict[str, List[Union[float, int]]],
              precision: int,
              colors: List[str],
              hatches: List[str] = None,
              single_width: float = 0.9,
              total_width: float = 0.8,
              legend_outside: bool = False,
              legend_location: str = None,
              label_rotation: str = None):
    bars = []

    num_bars = len(data)
    bar_width = total_width / num_bars

    for index, values in enumerate(data.values()):
        x_offset = (index - num_bars / 2) * bar_width + bar_width / 2

        for x, y in enumerate(values):
            bar = pyplot.bar(x + x_offset,
                             y,
                             width=bar_width * single_width,
                             color=colors[index % len(colors)],
                             hatch=hatches[index % len(hatches)] if hatches is not None else None,
                             alpha=0.99 if hatches is not None else 1,
                             zorder=2)

            if x == 0:
                bars.append(bar[0])

            annotate_bars(bar, precision, rotation=label_rotation)

    labels = data.keys()

    if legend_outside:
        pyplot.legend(bars,
                      labels,
                      fontsize=legend_font_size,
                      bbox_to_anchor=(1.04, 0.5),
                      loc="center left",
                      borderaxespad=0)
    elif legend_location is not None:
        pyplot.legend(bars, labels, fontsize=legend_font_size, loc=legend_location)
    else:
        pyplot.legend(bars, labels, fontsize=legend_font_size)


def plot_lines(x_values, y_values, colors, markers, labels):
    zorder = 2 + len(x_values)
    for index in range(len(x_values)):
        pyplot.plot(x_values[index],
                    y_values[index],
                    linestyle="-",
                    color=colors[index],
                    marker=markers[index],
                    markersize=4,
                    label=labels[index],
                    zorder=zorder)

        zorder -= 1


def plot_stacked_bars(segment_columns, segment_colors, segment_hatches, segment_labels, column, precision):
    handles = []

    for column_index, column_value in enumerate(data[column].tolist()):
        column_data = data[data[column] == column_value]

        bottom = 0
        for segment_index, segment_column in enumerate(segment_columns):
            bar = pyplot.bar(column_value,
                             column_data[segment_columns[segment_index]].tolist()[0],
                             bottom=bottom,
                             color=segment_colors[segment_index % len(segment_colors)],
                             hatch=segment_hatches[segment_index %
                                                   len(segment_hatches)] if segment_hatches is not None else None,
                             alpha=0.99 if segment_hatches is not None else 1,
                             label=segment_labels[segment_index] if column_index == 0 else "",
                             zorder=2)

            if column_index == 0:
                handles.append(bar)

            bottom += column_data[segment_columns[segment_index]].tolist()[0]

            if segment_index == len(segment_columns) - 1:
                annotate_bars(bar, precision, bottom)

    return handles


def configure_plot(x_ticks_ticks=None,
                   x_ticks_labels=None,
                   y_ticks_ticks=None,
                   y_ticks_labels=None,
                   x_ticks_rotation=False,
                   x_ticks_minor=False,
                   x_label=None,
                   y_label=None,
                   legend=False,
                   legend_location=None,
                   legend_handles=None,
                   legend_columns=None):

    pyplot.xticks(fontsize=small_font_size)
    if x_ticks_ticks is not None and x_ticks_labels is not None:
        pyplot.xticks(ticks=x_ticks_ticks, labels=x_ticks_labels)
    elif x_ticks_ticks is not None:
        pyplot.xticks(ticks=x_ticks_ticks)

    pyplot.yticks(fontsize=small_font_size)
    if y_ticks_ticks is not None and y_ticks_labels is not None:
        pyplot.yticks(ticks=y_ticks_ticks, labels=y_ticks_labels)
    elif y_ticks_ticks is not None:
        pyplot.yticks(ticks=y_ticks_ticks)

    if x_ticks_rotation:
        pyplot.xticks(rotation=45, ha="right")

    pyplot.minorticks_on()
    if not x_ticks_minor:
        pyplot.tick_params(axis='x', which='minor', bottom=False)

    if x_label is not None:
        pyplot.xlabel(x_label, fontsize=large_font_size)

    if y_label is not None:
        pyplot.ylabel(y_label, fontsize=large_font_size)

    if legend and legend_location is not None and legend_handles is not None and legend_columns is not None:
        pyplot.legend(fontsize=legend_font_size,
                      loc=legend_location,
                      handles=legend_handles,
                      ncol=legend_columns,
                      labelspacing=0.4)
    elif legend and legend_location is not None and legend_handles is not None:
        pyplot.legend(fontsize=legend_font_size, loc=legend_location, handles=legend_handles, labelspacing=0.4)
    elif legend and legend_location is not None:
        pyplot.legend(fontsize=legend_font_size, loc=legend_location, labelspacing=0.4)
    elif legend:
        pyplot.legend(fontsize=legend_font_size, labelspacing=0.4)


if __name__ == "__main__":
    settings.init()

    parser = argparse.ArgumentParser(description="plot multi-GPU experiments",
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=60))
    parser.add_argument(metavar="RUN", help="the run", action="store", type=str, default="", dest="run")
    parser.add_argument("-e",
                        "--experiments",
                        metavar="EXPERIMENTS",
                        help="the experiments",
                        action="store",
                        type=str,
                        default="",
                        dest="experiments")
    arguments = parser.parse_args()

    run = arguments.run
    experiments = list(filter(None, arguments.experiments.split(",")))

    script_path = pathlib.Path(__file__).parent.resolve()
    run_path = pathlib.Path(script_path / settings.experiments_path / run).resolve()

    for experiment in settings.experiments:
        if experiments and experiment.identifier not in experiments:
            continue

        try:
            data = pandas.read_csv(pathlib.Path(run_path / ("%s_%s.csv" %
                                                            (experiment.executable, experiment.identifier))).resolve(),
                                   header=0)
        except FileNotFoundError:
            continue

        ################################################################################################################
        # sort_benchmark
        ################################################################################################################

        if experiment.executable == "sort_benchmark":
            data = data.groupby([
                "num_elements", "algorithm", "gpus", "data_type", "distribution_type", "distribution_seed",
                "num_threads", "chunk_size", "merge_group_size", "num_buffers"
            ],
                                as_index=False).mean()

            max_gpus = max(data["gpus"].tolist(), key=len)
            num_gpus = max_gpus.count(",") + 1

            ############################################################################################################
            # sort_algorithm_to_sort_duration
            ############################################################################################################

            if experiment.identifier == "sort_algorithm_to_sort_duration":
                data = data[(data["num_elements"] == 4000000000)]

                gpus_configurations = []
                if num_gpus == 8:
                    gpus_configurations = ["0,2", "0,2,4,6"]
                elif num_gpus == 4:
                    gpus_configurations = ["0,1", "0,1,2,3"]

                algorithm_to_duration = {
                    "PARADIS (CPU)":
                        data[data["algorithm"] == "paradis"]["sort_duration"].tolist(),
                    "Thrust (1 GPU)":
                        data[(data["algorithm"] == "thrust") & (data["gpus"] == "0")]["sort_duration"].tolist(),
                    "P2P sort (2 GPUs)":
                        data[(data["algorithm"] == "tanasic") & (data["gpus"] == gpus_configurations[0])]
                        ["sort_duration"].tolist(),
                    "P2P sort (4 GPUs)":
                        data[(data["algorithm"] == "tanasic") & (data["gpus"] == gpus_configurations[1])]
                        ["sort_duration"].tolist(),
                    "HET sort (2 GPUs)":
                        data[(data["algorithm"] == "gowanlock") & (data["gpus"] == gpus_configurations[0])]
                        ["sort_duration"].tolist(),
                    "HET sort (4 GPUs)":
                        data[(data["algorithm"] == "gowanlock") & (data["gpus"] == gpus_configurations[1])]
                        ["sort_duration"].tolist(),
                }

                bar_colors = [colors[0], colors[1], colors[3], colors[3], colors[4], colors[4]]
                bar_hatches = [None, None, None, hatches[0], None, hatches[0]]

                scale_figure_size(2, 1)

                plot_bars(algorithm_to_duration, 2, bar_colors, bar_hatches, legend_outside=True)

                ticks = ["4"]

                y_ticks_ticks = []
                if num_gpus == 8:
                    y_ticks_ticks = numpy.arange(0, 3 + 0.5, 0.5)
                elif num_gpus == 4:
                    y_ticks_ticks = numpy.arange(0, 8 + 1, 2)

                configure_plot(x_ticks_ticks=range(len(ticks)),
                               x_ticks_labels=ticks,
                               y_ticks_ticks=y_ticks_ticks,
                               x_label="Number of keys [1e9]",
                               y_label="Sort duration [s]")

            ############################################################################################################
            # num_elements_to_tanasic_sort_duration_for_gpus
            ############################################################################################################

            elif experiment.identifier == "num_elements_to_tanasic_sort_duration_for_gpus":
                gpus_configurations = []
                if num_gpus == 8:
                    gpus_configurations = ["0", "0,2", "0,2,4,6", "0,1,2,3,4,5,6,7"]
                elif num_gpus == 4:
                    gpus_configurations = ["0", "0,1", "0,1,2,3"]
                elif num_gpus == 2:
                    gpus_configurations = ["0", "0,1"]

                for gpus_configuration in gpus_configurations:
                    data = data[(data["gpus"] != gpus_configuration) | (data["num_elements"] <= 2000000000 *
                                                                        (gpus_configuration.count(",") + 1))]

                data["num_elements"] = [num_elements / 1000000000 for num_elements in data["num_elements"]]

                x_values = [data[data["gpus"] == gpus]["num_elements"].tolist() for gpus in gpus_configurations]
                y_values = [data[data["gpus"] == gpus]["sort_duration"].tolist() for gpus in gpus_configurations]

                line_colors = [colors[0], colors[4], colors[1], colors[3], colors[2]]
                line_markers = ["o", "v", "s", "d", "h"]

                line_labels = []
                for gpus_configuration in gpus_configurations:
                    num_gpus = gpus_configuration.count(",") + 1
                    line_labels.append("%s GPU%s" % (num_gpus, "s" if num_gpus > 1 else ""))

                scale_figure_size(1, 1)

                plot_lines(x_values, y_values, line_colors, line_markers, line_labels)

                configure_plot(y_ticks_ticks=numpy.arange(0, 3 + 1, 1),
                               x_label="Number of keys [1e9]",
                               y_label="Sort duration [s]",
                               legend=True,
                               legend_location="upper left")

            ############################################################################################################
            # num_elements_to_gowanlock_sort_duration_for_gpus
            ############################################################################################################

            elif experiment.identifier == "num_elements_to_gowanlock_sort_duration_for_gpus":
                gpus_configurations = []
                if num_gpus == 8:
                    gpus_configurations = ["0", "0,2", "0,2,4,6", "0,1,2,3,4,5,6,7"]
                elif num_gpus == 4:
                    gpus_configurations = ["0", "0,1", "0,1,2,3"]
                elif num_gpus == 2:
                    gpus_configurations = ["0", "0,1"]

                for gpus_configuration in gpus_configurations:
                    data = data[(data["gpus"] != gpus_configuration) | (data["num_elements"] <= 2000000000 *
                                                                        (gpus_configuration.count(",") + 1))]

                data["num_elements"] = [num_elements / 1000000000 for num_elements in data["num_elements"]]

                x_values = [data[data["gpus"] == gpus]["num_elements"].tolist() for gpus in gpus_configurations]
                y_values = [data[data["gpus"] == gpus]["sort_duration"].tolist() for gpus in gpus_configurations]

                line_colors = [colors[0], colors[4], colors[1], colors[3], colors[2]]
                line_markers = ["o", "v", "s", "d", "h"]

                line_labels = []
                for gpus_configuration in gpus_configurations:
                    num_gpus = gpus_configuration.count(",") + 1
                    line_labels.append("%s GPU%s" % (num_gpus, "s" if num_gpus > 1 else ""))

                scale_figure_size(1, 1)

                plot_lines(x_values, y_values, line_colors, line_markers, line_labels)

                configure_plot(y_ticks_ticks=numpy.arange(0, 3 + 1, 1),
                               x_label="Number of keys [1e9]",
                               y_label="Sort duration [s]",
                               legend=True,
                               legend_location="upper left")

            ############################################################################################################
            # large_data_to_gowanlock_sort_duration_for_gpus
            ############################################################################################################

            elif experiment.identifier == "large_data_to_gowanlock_sort_duration_for_gpus":
                data["num_elements"] = [num_elements / 1000000000 for num_elements in data["num_elements"]]

                gpus_configuration = ""
                if num_gpus == 8:
                    gpus_configuration = "0,1,2,3,4,5,6,7"
                elif num_gpus == 4:
                    gpus_configuration = "0,1,2,3"
                elif num_gpus == 2:
                    gpus_configuration = "0,1"

                x_values = []
                y_values = []
                for num_buffers in [2, 3]:
                    for merge_group_size in [1, 0]:
                        x_values.append(
                            data[(data["gpus"] == gpus_configuration) & (data["num_buffers"] == num_buffers) &
                                 (data["merge_group_size"] == merge_group_size)]["num_elements"].tolist())
                        y_values.append(
                            data[(data["gpus"] == gpus_configuration) & (data["num_buffers"] == num_buffers) &
                                 (data["merge_group_size"] == merge_group_size)]["sort_duration"].tolist())

                line_colors = [colors[0], colors[4], colors[1], colors[3]]
                line_markers = ["o", "v", "s", "d"]

                line_labels = ["3n", "3n + EM", "2n", "2n + EM"]

                scale_figure_size(1, 1)

                plot_lines(x_values, y_values, line_colors, line_markers, line_labels)

                y_ticks_ticks = []
                if num_gpus == 8:
                    y_ticks_ticks = numpy.arange(0, 25 + 1, 5)
                elif num_gpus == 4 or num_gpus == 2:
                    y_ticks_ticks = numpy.arange(0, 60 + 1, 10)

                configure_plot(y_ticks_ticks=y_ticks_ticks,
                               x_label="Number of keys [1e9]",
                               y_label="Sort duration [s]",
                               legend=True,
                               legend_location="upper left")

            ############################################################################################################
            # large_data_to_gowanlock_vs_cpu_sort_duration_for_gpus
            ############################################################################################################

            elif experiment.identifier == "large_data_to_gowanlock_vs_cpu_sort_duration_for_gpus":
                data["num_elements"] = [num_elements / 1000000000 for num_elements in data["num_elements"]]

                gpus_configuration = ""
                if num_gpus == 8:
                    gpus_configuration = "0,1,2,3,4,5,6,7"
                elif num_gpus == 4:
                    gpus_configuration = "0,1,2,3"
                elif num_gpus == 2:
                    gpus_configuration = "0,1"

                data = data[(data["algorithm"].isin(["paradis"])) | (data["algorithm"] == "gowanlock") &
                            (data["gpus"] == gpus_configuration)]

                x_values = []
                y_values = []
                for algorithm in ["paradis", "gowanlock"]:
                    x_values.append(data[data["algorithm"] == algorithm]["num_elements"].tolist())
                    y_values.append(data[data["algorithm"] == algorithm]["sort_duration"].tolist())

                line_colors = [colors[4], colors[0], colors[1]]
                line_markers = ["o", "v", "s"]

                line_labels = ["PARADIS (CPU)", "HET sort (%s GPUs)" % (gpus_configuration.count(",") + 1)]

                scale_figure_size(1, 1)

                plot_lines(x_values, y_values, line_colors, line_markers, line_labels)

                y_ticks_ticks = []
                if num_gpus == 8:
                    y_ticks_ticks = numpy.arange(0, 35 + 1, 5)
                elif num_gpus == 4 or num_gpus == 2:
                    y_ticks_ticks = numpy.arange(0, 100 + 1, 20)

                configure_plot(y_ticks_ticks=y_ticks_ticks,
                               x_label="Number of keys [1e9]",
                               y_label="Sort duration [s]",
                               legend=True,
                               legend_location="upper left")

            ############################################################################################################
            # distribution_type_to_sort_duration_for_algorithm
            ############################################################################################################

            elif experiment.identifier == "distribution_type_to_sort_duration_for_algorithm":
                if num_gpus == 8:
                    gpus_configuration = "0,2,4,6"
                elif num_gpus == 4:
                    gpus_configuration = "0,1"
                elif num_gpus == 2:
                    gpus_configuration = "0,1"

                algorithms = ["tanasic", "gowanlock"]
                distribution_types = ["uniform", "normal", "sorted", "reverse-sorted", "nearly-sorted"]

                data = data[(data["distribution_type"].isin(distribution_types)) & (data["gpus"] == gpus_configuration)]

                data["algorithm"] = pandas.Categorical(data["algorithm"], categories=algorithms, ordered=True)
                data = data.sort_values("algorithm", ascending=True)

                data["distribution_type"] = pandas.Categorical(data["distribution_type"],
                                                               categories=distribution_types,
                                                               ordered=True)
                data = data.sort_values("distribution_type", ascending=True)

                distribution_type_to_durations = {}
                for distribution_type in distribution_types:
                    distribution_type_to_durations[distribution_type.capitalize()] = data[
                        data["distribution_type"] == distribution_type]["sort_duration"].tolist()

                scale_figure_size(2, 1)

                plot_bars(distribution_type_to_durations, 2, colors, legend_outside=True, label_rotation="vertical")

                ticks = ["P2P sort", "HET sort"]

                y_ticks_ticks = []
                if num_gpus == 8:
                    y_ticks_ticks = numpy.arange(0, 0.7, 0.1)
                elif num_gpus == 4 or num_gpus == 2:
                    y_ticks_ticks = numpy.arange(0, 0.6, 0.1)

                configure_plot(x_ticks_ticks=range(len(ticks)),
                               x_ticks_labels=ticks,
                               y_ticks_ticks=y_ticks_ticks,
                               x_label="Distribution type",
                               y_label="Sort duration [s]")

            ############################################################################################################
            # gpus_to_tanasic_sort_duration_profilers_nsys
            ############################################################################################################

            elif experiment.identifier == "gpus_to_tanasic_sort_duration_profilers_nsys":
                gpus_configurations = []
                if num_gpus == 8:
                    gpus_configurations = ["0", "0,2", "0,2,4,6", "0,1,2,3,4,5,6,7"]
                elif num_gpus == 4:
                    gpus_configurations = ["0", "0,1", "0,1,2,3"]
                elif num_gpus == 2:
                    gpus_configurations = ["0", "0,1"]

                data = data[data["gpus"].isin(gpus_configurations)]

                data["gpus"] = pandas.Categorical(data["gpus"], categories=gpus_configurations, ordered=True)
                data = data.sort_values("gpus", ascending=True)

                data = data.groupby(["gpus"], as_index=False).mean()

                segment_columns = [
                    "h_to_d_total_duration", "sort_total_duration", "merge_total_duration", "d_to_h_total_duration"
                ]
                segment_colors = [colors[0], colors[4], colors[3], colors[1]]
                segment_hatches = [hatches[0], None, None, hatches[1]]
                segment_labels = ["HtoD", "Sort", "Merge", "DtoH"]

                scale_figure_size(1, 1)

                handles = plot_stacked_bars(segment_columns, segment_colors, segment_hatches, segment_labels, "gpus", 2)

                ticks = [gpus_configuration.count(",") + 1 for gpus_configuration in gpus_configurations]

                y_ticks_ticks = []
                if num_gpus == 8:
                    y_ticks_ticks = numpy.arange(0, 1.5, 0.2)
                elif num_gpus == 4 or num_gpus == 2:
                    y_ticks_ticks = numpy.arange(0, 0.9, 0.2)

                configure_plot(x_ticks_ticks=range(len(ticks)),
                               x_ticks_labels=ticks,
                               y_ticks_ticks=y_ticks_ticks,
                               x_label="Number of GPUs",
                               y_label="Sort duration [s]",
                               legend=True,
                               legend_location="upper right",
                               legend_handles=handles[::-1],
                               legend_columns=2)

            ############################################################################################################
            # gpus_to_gowanlock_sort_duration_profilers_nsys
            ############################################################################################################

            elif experiment.identifier == "gpus_to_gowanlock_sort_duration_profilers_nsys":
                gpus_configurations = []
                if num_gpus == 8:
                    gpus_configurations = ["0", "0,2", "0,2,4,6", "0,1,2,3,4,5,6,7"]
                elif num_gpus == 4:
                    gpus_configurations = ["0", "0,1", "0,1,2,3"]
                elif num_gpus == 2:
                    gpus_configurations = ["0", "0,1"]

                data = data[data["gpus"].isin(gpus_configurations)]

                data["gpus"] = pandas.Categorical(data["gpus"], categories=gpus_configurations, ordered=True)
                data = data.sort_values("gpus", ascending=True)

                data = data.groupby(["gpus"], as_index=False).mean()

                segment_columns = [
                    "h_to_d_total_duration", "sort_total_duration", "merge_total_duration", "d_to_h_total_duration"
                ]
                segment_colors = [colors[0], colors[4], colors[3], colors[1]]
                segment_hatches = [hatches[0], None, None, hatches[1]]
                segment_labels = ["HtoD", "Sort", "Merge", "DtoH"]

                scale_figure_size(1, 1)

                handles = plot_stacked_bars(segment_columns, segment_colors, segment_hatches, segment_labels, "gpus", 2)

                ticks = [gpus_configuration.count(",") + 1 for gpus_configuration in gpus_configurations]

                y_ticks_ticks = []
                if num_gpus == 8:
                    y_ticks_ticks = numpy.arange(0, 1.5, 0.2)
                elif num_gpus == 4 or num_gpus == 2:
                    y_ticks_ticks = numpy.arange(0, 0.9, 0.2)

                configure_plot(x_ticks_ticks=range(len(ticks)),
                               x_ticks_labels=ticks,
                               y_ticks_ticks=y_ticks_ticks,
                               x_label="Number of GPUs",
                               y_label="Sort duration [s]",
                               legend=True,
                               legend_location="upper right",
                               legend_handles=handles[::-1],
                               legend_columns=2)

            else:
                continue

        ################################################################################################################
        # data_transfer_benchmark
        ################################################################################################################

        elif experiment.executable == "data_transfer_benchmark":
            data = data.groupby(["num_bytes", "gpus", "execution_type"], as_index=False).agg({
                "h_to_d_durations": calculate_means,
                "h_to_d_total_duration": "mean",
                "p_to_p_durations": calculate_means,
                "p_to_p_total_duration": "mean",
                "d_to_h_durations": calculate_means,
                "d_to_h_total_duration": "mean",
                "bidirectional_durations": calculate_means,
                "total_bidirectional_duration": "mean"
            })

            num_gigabytes = data["num_bytes"].tolist()[0] / 1000000000

            max_gpus = max(data["gpus"].tolist(), key=len)
            num_gpus = max_gpus.count(",") + 1

            ############################################################################################################
            # num_bytes_to_h_to_d_total_duration_and_d_to_h_total_duration
            ############################################################################################################

            if experiment.identifier == "num_bytes_to_h_to_d_total_duration_and_d_to_h_total_duration":
                data = data[data["gpus"] == max_gpus]

                direction_to_durations = {
                    "HtoD": [
                        num_gigabytes / data["h_to_d_durations"].tolist()[0][0][0],
                        num_gigabytes / data["h_to_d_durations"].tolist()[0][0][num_gpus // 2]
                    ],
                    "DtoH": [
                        num_gigabytes / data["d_to_h_durations"].tolist()[0][0][0],
                        num_gigabytes / data["d_to_h_durations"].tolist()[0][0][num_gpus // 2]
                    ],
                    "HtoD/DtoH": [
                        2 * num_gigabytes / data["bidirectional_durations"].tolist()[0][0][0],
                        2 * num_gigabytes / data["bidirectional_durations"].tolist()[0][0][num_gpus // 2]
                    ]
                }

                scale_figure_size(1, 1)

                plot_bars(direction_to_durations, 0, colors, hatches, label_rotation="vertical")

                ticks = []
                if num_gpus == 8:
                    ticks = ["{0–3}", "{4–7}"]
                elif num_gpus == 4:
                    ticks = ["{0, 1}", "{2, 3}"]
                elif num_gpus == 2:
                    ticks = ["0", "1"]

                configure_plot(x_ticks_ticks=range(len(ticks)),
                               x_ticks_labels=ticks,
                               y_ticks_ticks=numpy.arange(0, 250 + 1, 50),
                               x_label="GPU",
                               y_label="Throughput [GB/s]")

            ############################################################################################################
            # num_bytes_to_p_to_p_total_duration
            ############################################################################################################

            elif experiment.identifier == "num_bytes_to_p_to_p_total_duration":
                direction_to_durations = {}
                if num_gpus == 8:
                    direction_to_durations["PtoP"] = [num_gigabytes / data["p_to_p_durations"].tolist()[0][0][0]]
                elif num_gpus == 4:
                    direction_to_durations["PtoP"] = [
                        num_gigabytes / duration for duration in data["p_to_p_durations"].tolist()[0][0]
                    ]
                elif num_gpus == 2:
                    direction_to_durations["PtoP"] = [num_gigabytes / data["p_to_p_durations"].tolist()[0][0][0]]

                scale_figure_size(1, 1)

                plot_bars(direction_to_durations, 0, [colors[3]], [hatches[3]])

                ticks = []
                if num_gpus == 8:
                    ticks = ["i$\\rightarrow$j"]
                elif num_gpus == 4:
                    ticks = ["0$\\rightarrow$1", "0$\\rightarrow$2", "0$\\rightarrow$3"]
                elif num_gpus == 2:
                    ticks = ["0$\\rightarrow$1"]

                y_ticks_ticks = []
                if num_gpus == 8:
                    y_ticks_ticks = numpy.arange(0, 2500 + 1, 500)
                elif num_gpus == 4 or num_gpus == 2:
                    y_ticks_ticks = numpy.arange(0, 250 + 1, 50)

                configure_plot(x_ticks_ticks=range(len(ticks)),
                               x_ticks_labels=ticks,
                               y_ticks_ticks=y_ticks_ticks,
                               x_label="GPU$\\rightarrow$GPU",
                               y_label="Throughput [GB/s]")

            ############################################################################################################
            # gpus_to_h_to_d_total_duration_and_d_to_h_total_duration
            ############################################################################################################

            elif experiment.identifier == "gpus_to_h_to_d_total_duration_and_d_to_h_total_duration":
                direction_to_durations = {"HtoD": [], "DtoH": [], "HtoD/DtoH": []}

                if num_gpus == 8:
                    serial_data = data[(data["execution_type"] == "serial") & (data["gpus"] == "0,1,2,3,4,5,6,7")]

                    for index in [0, 4]:
                        direction_to_durations["HtoD"].append(num_gigabytes /
                                                              serial_data["h_to_d_durations"].tolist()[0][0][index])
                        direction_to_durations["DtoH"].append(num_gigabytes /
                                                              serial_data["d_to_h_durations"].tolist()[0][0][index])
                        direction_to_durations["HtoD/DtoH"].append(
                            2 * num_gigabytes / serial_data["bidirectional_durations"].tolist()[0][0][index])

                data = data[data["execution_type"] == "parallel"]

                gpus_configurations = []
                if num_gpus == 8:
                    gpus_configurations = ["0,1", "0,2", "4,6", "0,2,4,6", "0,1,2,3,4,5,6,7"]
                elif num_gpus == 4:
                    gpus_configurations = ["0,1", "2,3", "0,1,2,3"]
                elif num_gpus == 2:
                    gpus_configurations = ["0,1"]

                for gpus in gpus_configurations:
                    total_num_gigabytes = (gpus.count(",") + 1) * num_gigabytes

                    direction_to_durations["HtoD"].append(
                        total_num_gigabytes / data[data["gpus"] == gpus]["h_to_d_total_duration"].tolist()[0])
                    direction_to_durations["DtoH"].append(
                        total_num_gigabytes / data[data["gpus"] == gpus]["d_to_h_total_duration"].tolist()[0])
                    direction_to_durations["HtoD/DtoH"].append(
                        2 * total_num_gigabytes /
                        data[data["gpus"] == gpus]["total_bidirectional_duration"].tolist()[0])

                legend_location = ""
                if num_gpus == 8:
                    legend_location = "upper left"

                    scale_figure_size(2, 1)
                elif num_gpus == 4 or num_gpus == 2:
                    legend_location = "upper right"

                    scale_figure_size(1, 1)

                plot_bars(direction_to_durations,
                          0,
                          colors,
                          hatches,
                          legend_location=legend_location,
                          label_rotation="vertical")

                ticks = []

                if num_gpus == 8:
                    ticks.extend(["{0–3}", "{4–7}"])

                    gpus_configurations[-1] = "0–7"

                for gpus in gpus_configurations:
                    ticks.append("(%s)" % (gpus.replace(",", ", ")))

                configure_plot(x_ticks_ticks=range(len(ticks)),
                               x_ticks_labels=ticks,
                               y_ticks_ticks=numpy.arange(0, 250 + 1, 50),
                               x_label="GPUs",
                               y_label="Throughput [GB/s]")

            ############################################################################################################
            # gpus_to_p_to_p_total_duration
            ############################################################################################################

            elif experiment.identifier == "gpus_to_p_to_p_total_duration":
                direction_to_durations = {"PtoP": []}

                if num_gpus == 8:
                    serial_data = data[(data["execution_type"] == "serial") & (data["gpus"] == "0,1,2,3,4,5,6,7")]

                    direction_to_durations["PtoP"].append(num_gigabytes /
                                                          serial_data["p_to_p_durations"].tolist()[0][0][0])

                data = data[data["execution_type"] == "parallel"]

                gpus_configurations = []
                if num_gpus == 8:
                    gpus_configurations = ["0,1", "0,2", "0,2,4,6", "0,1,2,3", "0,1,2,3,4,5,6,7"]
                elif num_gpus == 4:
                    gpus_configurations = ["0,1", "2,3", "0,1,2,3"]
                elif num_gpus == 2:
                    gpus_configurations = ["0,1"]

                for gpus in gpus_configurations:
                    total_num_gigabytes = (gpus.count(",") + 1) * num_gigabytes

                    direction_to_durations["PtoP"].append(
                        total_num_gigabytes / data[data["gpus"] == gpus]["p_to_p_total_duration"].tolist()[0])

                legend_location = ""
                if num_gpus == 8:
                    legend_location = "upper left"

                    scale_figure_size(2, 1)
                elif num_gpus == 4 or num_gpus == 2:
                    legend_location = "upper right"

                    scale_figure_size(1, 1)

                plot_bars(direction_to_durations, 0, [colors[3]], [hatches[3]], legend_location=legend_location)

                ticks = []
                if num_gpus == 8:
                    ticks = [
                        "i$\\rightarrow$j", "0$\leftrightarrow$1", "0$\leftrightarrow$2",
                        "0$\leftrightarrow$6,\n2$\leftrightarrow$4,", "0$\leftrightarrow$3,\n1$\leftrightarrow$2,",
                        "0$\leftrightarrow$7, $\ldots$,\n3$\leftrightarrow$4,"
                    ]
                elif num_gpus == 4:
                    ticks = ["0$\leftrightarrow$1", "2$\leftrightarrow$3", "0$\leftrightarrow$3, 1$\leftrightarrow$2"]
                elif num_gpus == 2:
                    ticks = ["0$\leftrightarrow$1"]

                y_ticks_ticks = []
                if num_gpus == 8:
                    y_ticks_ticks = numpy.arange(0, 2500 + 1, 500)
                elif num_gpus == 4 or num_gpus == 2:
                    y_ticks_ticks = numpy.arange(0, 250 + 1, 50)

                x_label = ""
                if num_gpus == 8:
                    x_label = "GPU$\\rightarrow$/$\leftrightarrow$GPU"
                elif num_gpus == 4 or num_gpus == 2:
                    x_label = "GPU$\leftrightarrow$GPU"

                configure_plot(x_ticks_ticks=range(len(ticks)),
                               x_ticks_labels=ticks,
                               y_ticks_ticks=y_ticks_ticks,
                               x_label=x_label,
                               y_label="Throughput [GB/s]")

            else:
                continue

        else:
            continue

        pyplot.tight_layout()
        pyplot.savefig(pathlib.Path(run_path / ("%s_%s.pdf" %
                                                (experiment.executable, experiment.identifier))).resolve(),
                       format="pdf")
        pyplot.close()
