from random import sample
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as pltc
from matplotlib.collections import LineCollection
all_colors = [k for k, v in pltc.cnames.items()]


def plot_histogram_from_hdf5(hdf5_dataset):
    data = hdf5_dataset.labels_dset
    fig1, ax1 = plt.subplots(3, sharex=True)
    fig1.suptitle('GT position hist', fontsize=50)
    ax1[0].hist(data[:, 0], bins=1000, density=False)
    ax1[0].grid(True)
    ax1[1].hist(data[:, 1], bins=1000, density=False)
    ax1[1].grid(True)
    ax1[2].hist(data[:, 2], bins=1000, density=False)
    ax1[2].grid(True)
    plt.show()


def plot_histogram_from_array(dataset, index):
    plt.hist(dataset[:, index], density=False, bins=1000)
    plt.ylabel('Number of samples')
    plt.xlabel('X GT delta poses')
    plt.show()


def plot_histogram(dataset):
    variables_values = return_individual_variables(dataset)
    plt.hist(variables_values["i"], density=False, bins=1000)
    plt.ylabel('Number of samples')
    plt.xlabel('X GT delta poses')
    plt.show()


def prepare_variables_col_indices(input_dataset):
    # Prepare indices for all variables in input.
    variables_col_indices = {}
    all_indices = np.arange(input_dataset.examples_dset.shape[1])
    variables = ["x", "y", "z", "v", "i"]
    for i, variable in enumerate(variables, 0):
        variables_col_indices[variable] = all_indices[i::5]
    return variables_col_indices


def return_individual_variables(input_dataset):
    variables_col_indices = prepare_variables_col_indices(input_dataset)
    variables_values = {"x": np.empty(shape=(0, 1), dtype=np.float64), "y": np.empty(shape=(0, 1), dtype=np.float64),
                        "z": np.empty(shape=(0, 1), dtype=np.float64), "v": np.empty(shape=(0, 1), dtype=np.float64),
                        "i": np.empty(shape=(0, 1), dtype=np.float64)}
    for variable, indices in variables_col_indices.items():
        variables_values[variable] = np.append(
            variables_values[variable], input_dataset.examples_dset[:, indices])
    return variables_values


def plot_gt_from_norm(gt_debug, gt_debug_time):
    fig, ax = plt.subplots(3)
    fig.suptitle('GT position', fontsize=50)
    plot, = ax[0].plot(gt_debug_time,
                       np.cumsum(gt_debug[:, 0]), 'r*', label="gt pose")
    ax[0].grid(True)
    plot, = ax[1].plot(gt_debug_time, np.cumsum(gt_debug[:, 1]), 'r*')
    ax[1].grid(True)
    plot, = ax[2].plot(gt_debug_time, np.cumsum(gt_debug[:, 2]), 'r*')
    ax[2].grid(True)
    fig.align_ylabels(ax)

    fig1, ax1 = plt.subplots(3)
    fig1.suptitle('GT position hist', fontsize=50)
    ax1[0].hist(gt_debug[:, 0], bins=1000, density=False)
    ax1[0].grid(True)
    ax1[1].hist(gt_debug[:, 1], bins=1000, density=False)
    ax1[1].grid(True)
    ax1[2].hist(gt_debug[:, 2], bins=1000, density=False)
    ax1[2].grid(True)

    plt.show()


def plot_gt_from_norm_nocumsum(gt_debug, gt_debug_time):
    fig1, ax1 = plt.subplots(3)
    fig1.suptitle('GT position hist', fontsize=50)
    ax1[0].hist(gt_debug[:, 0], bins=1000, density=False)
    ax1[0].grid(True)
    ax1[1].hist(gt_debug[:, 1], bins=1000, density=False)
    ax1[1].grid(True)
    ax1[2].hist(gt_debug[:, 2], bins=1000, density=False)
    ax1[2].grid(True)

    plt.show()


def plot_gt_from_hdf5(hdf5_dataset):
    data = hdf5_dataset.labels_dset
    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle('GT position', fontsize=50)
    plot, = ax[0].plot(
        np.cumsum(data[:, 0]), 'r', label="acc bias")
    ax[0].grid(True)
    plot, = ax[1].plot(np.cumsum(data[:, 1]), 'r')
    ax[1].grid(True)
    plot, = ax[2].plot(np.cumsum(data[:, 2]), 'r')
    ax[2].grid(True)
    fig.align_ylabels(ax)
    plt.show()


def plot_gt_from_array(data):
    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle('GT position', fontsize=50)
    plot, = ax[0].plot(
        np.cumsum(data[:, 0]), 'r', label="acc bias")
    ax[0].grid(True)
    plot, = ax[1].plot(np.cumsum(data[:, 1]), 'r')
    ax[1].grid(True)
    plot, = ax[2].plot(np.cumsum(data[:, 2]), 'r')
    ax[2].grid(True)
    fig.align_ylabels(ax)
    plt.show()


def plot_gt_interp_and_not(
        gt_delta_time, gt_delta_interp, gt_time, gt_position, coeffs, window_length):
    # Plot ground truth along with the interpolated ground truth data.
    lines_x = np.empty(shape=(0, 2), dtype=np.float32)
    lines_y = np.empty(shape=(0, 2), dtype=np.float32)
    lines_z = np.empty(shape=(0, 2), dtype=np.float32)
    residuals = np.empty(shape=(0, 3), dtype=np.float32)
    for coeff in coeffs:
        for t in coeff[1]:
            lines_x = np.append(lines_x, np.asarray(
                [t, coeff[0][0, 0]*t + coeff[0][1, 0]],
                dtype=np.float32)[np.newaxis, :], axis=0)
            lines_y = np.append(lines_y, np.asarray(
                [t, coeff[0][0, 1]*t + coeff[0][1, 1]],
                dtype=np.float32)[np.newaxis, :], axis=0)
            lines_z = np.append(lines_z, np.asarray(
                [t, coeff[0][0, 2]*t + coeff[0][1, 2]],
                dtype=np.float32)[np.newaxis, :], axis=0)
        residuals = np.append(residuals, np.asarray(
            np.sqrt(coeff[2]),
            dtype=np.float32)[np.newaxis, :], axis=0)
    lines_x = [lines_x]
    lines_y = [lines_y]
    lines_z = [lines_z]

    fig, ax = plt.subplots(3)
    fig.suptitle('GT position', fontsize=50)
    plot, = ax[0].plot(gt_delta_time[:-window_length-1],
                       np.cumsum(gt_delta_interp[:, 0]), 'r*', label="gt interp")
    plot, = ax[0].plot(gt_time,
                       gt_position[:, 0], 'b+', label="gt no-interp")
    ax[0].grid(True)
    line_collection_x = LineCollection(lines_x)
    ax[0].add_collection(line_collection_x)

    plot, = ax[1].plot(gt_delta_time[:-window_length-1],
                       np.cumsum(gt_delta_interp[:, 1]), 'r*')
    plot, = ax[1].plot(gt_time,
                       gt_position[:, 1], 'b+', label="gt no-interp")
    ax[1].grid(True)
    line_collection_y = LineCollection(lines_y)
    ax[1].add_collection(line_collection_y)

    plot, = ax[2].plot(gt_delta_time[:-window_length-1],
                       np.cumsum(gt_delta_interp[:, 2]), 'r*')
    plot, = ax[2].plot(gt_time,
                       gt_position[:, 2], 'b+', label="gt no-interp")
    ax[2].grid(True)
    line_collection_z = LineCollection(lines_z)
    ax[2].add_collection(line_collection_z)

    fig.align_ylabels(ax)

    # deltas
    fig0, ax0 = plt.subplots(3)
    fig0.suptitle('deltas', fontsize=50)
    plot, = ax0[0].plot(gt_delta_time[:-window_length-1],
                        gt_delta_interp[:, 0], 'r*', label="gt deltas")
    ax0[0].grid(True)
    plot, = ax0[1].plot(gt_delta_time[:-window_length-1],
                        gt_delta_interp[:, 1], 'r*')
    ax0[1].grid(True)
    plot, = ax0[2].plot(gt_delta_time[:-window_length-1],
                        gt_delta_interp[:, 2], 'r*')
    ax0[2].grid(True)
    fig0.align_ylabels(ax)

    # Resiudals
    fig1, ax1 = plt.subplots(3)
    fig1.suptitle('RMSE of the affine fit', fontsize=50)
    plot, = ax1[0].plot(gt_delta_time[:-window_length],
                        residuals[:, 0], 'r*', label="rmse")
    plot, = ax1[0].plot(gt_delta_time[:-window_length-1],
                        gt_delta_interp[:, 0], 'g', label="gt deltas")
    ax1[0].grid(True)
    ax1[0].set_ylabel('fit rmse [m]',  fontsize=50)
    plot, = ax1[1].plot(gt_delta_time[:-window_length],
                        residuals[:, 1], 'r*')
    plot, = ax1[1].plot(gt_delta_time[:-window_length-1],
                        gt_delta_interp[:, 1], 'g')
    ax1[1].grid(True)
    plot, = ax1[2].plot(gt_delta_time[:-window_length],
                        residuals[:, 2], 'r*')
    plot, = ax1[2].plot(gt_delta_time[:-window_length-1],
                        gt_delta_interp[:, 2], 'g')
    ax1[2].grid(True)
    fig1.align_ylabels(ax1)

    plt.show()


def plot_gt_interp_and_not_hist(
        gt_position, gt_delta_interp):
    gt_delta = np.diff(gt_position, axis=0)
    # Plot ground truth along with the interpolated ground truth data
    # as histograms.
    fig1, ax1 = plt.subplots(3, sharex=True)
    fig1.suptitle('GT position hist', fontsize=50)
    ax1[0].hist(gt_delta[:, 0], bins=1000,
                density=False, color='b', label='meas.')
    ax1[0].hist(gt_delta_interp[:, 0], bins=1000,
                density=False, color='g', label='interp.')
    ax1[0].grid(True)
    ax1[1].hist(gt_delta[:, 1], bins=1000,
                density=False, color='b', label='meas.')
    ax1[1].hist(gt_delta_interp[:, 1], bins=1000,
                density=False, color='g', label='interp.')
    ax1[1].grid(True)
    ax1[2].hist(gt_delta[:, 2], bins=1000,
                density=False, color='b', label='meas.')
    ax1[2].hist(gt_delta_interp[:, 2], bins=1000,
                density=False, color='g', label='interp.')
    ax1[2].grid(True)
    plt.show()


def plot_gt_interp_and_measured(
        gt_time, gt_position, interp):

    gt_delta = np.diff(gt_position, axis=0)
    gt_interp = interp(gt_time)
    gt_delta_interp = np.diff(gt_interp, axis=0)
    # Plot ground truth along with the interpolated ground truth data
    # as histograms.
    fig1, ax1 = plt.subplots(3, 2, sharex=True)
    fig1.suptitle('GT delta position hist', fontsize=50)
    ax1[0, 0].hist(gt_delta[:, 0], bins=1000,
                   density=False, color='b', label='meas.')
    ax1[0, 1].hist(gt_delta_interp[:, 0], bins=1000,
                   density=False, color='g', label='interp.')
    ax1[0, 0].set_ylabel('delta x', fontsize=50)
    ax1[0, 0].grid(True)
    ax1[0, 1].grid(True)
    ax1[0, 1].legend()
    ax1[0, 0].legend()
    ax1[1, 0].hist(gt_delta[:, 1], bins=1000,
                   density=False, color='b', label='meas.')
    ax1[1, 1].hist(gt_delta_interp[:, 1], bins=1000,
                   density=False, color='g', label='interp.')
    ax1[1, 0].set_ylabel('delta y', fontsize=50)
    ax1[1, 0].grid(True)
    ax1[1, 1].grid(True)
    ax1[2, 0].hist(gt_delta[:, 2], bins=1000,
                   density=False, color='b', label='meas.')
    ax1[2, 1].hist(gt_delta_interp[:, 2], bins=1000,
                   density=False, color='g', label='interp.')
    ax1[2, 0].set_ylabel('delta z', fontsize=50)
    ax1[2, 0].grid(True)
    ax1[2, 1].grid(True)

    fig2, ax2 = plt.subplots(3, sharex=True)
    fig2.suptitle('GT meas with interp diff hist', fontsize=50)
    ax2[0].hist(gt_delta[:, 0] - gt_delta_interp[:, 0], bins=1000,
                density=False, color='b', label='meas.')
    ax2[0].set_ylabel('x diff', fontsize=50)
    ax2[0].grid(True)
    ax2[1].hist(gt_delta[:, 1] - gt_delta_interp[:, 1], bins=1000,
                density=False, color='b', label='meas.')
    ax2[1].set_ylabel('y diff', fontsize=50)
    ax2[1].grid(True)
    ax2[2].hist(gt_delta[:, 2] - gt_delta_interp[:, 2], bins=1000,
                density=False, color='g', label='meas.')
    ax2[2].set_ylabel('z diff', fontsize=50)
    ax2[2].grid(True)

    plt.show()