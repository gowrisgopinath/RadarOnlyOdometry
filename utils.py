import os 
from pathlib import Path
import struct
from pypcd4 import PointCloud
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R, Slerp

def get_doppler_rcs_minmax(pcls):
    doppler_min, doppler_max = np.inf, -np.inf
    rcs_min, rcs_max = np.inf, -np.inf

    for pc_path in pcls:
        try:
            pc = PointCloud.from_path(pc_path)
            doppler = pc.pc_data['doppler']
            intensity = pc.pc_data['intensity']

            doppler = doppler[np.isfinite(doppler)]
            intensity = intensity[np.isfinite(intensity)]

            if doppler.size > 0:
                d_min, d_max = np.min(doppler), np.max(doppler)
                doppler_min = min(doppler_min, d_min)
                doppler_max = max(doppler_max, d_max)

            if intensity.size > 0:
                i_min, i_max = np.min(intensity), np.max(intensity)
                rcs_min = min(rcs_min, i_min)
                rcs_max = max(rcs_max, i_max)

        except Exception as e:
            print(f"Skipping {pc_path}: {e}")
            continue

    doppler_stats = {'min': float(doppler_min), 'max': float(doppler_max)}
    rcs_stats = {'min': float(rcs_min), 'max': float(rcs_max)}

    print(f"Global Doppler range: {doppler_stats['min']:.3f} -> {doppler_stats['max']:.3f}")
    print(f"Global RCS range: {rcs_stats['min']:.3f} -> {rcs_stats['max']:.3f}")
    return doppler_stats, rcs_stats

def get_sequences(path):
    sequences = [os.path.join(path, d) for d in os.listdir(path)]
    sequences = [d for d in sequences if os.path.isdir(d)]
    sequences.sort()
    return sequences

def get_pcls_from_path(path):
    pcls = []
    seq_paths = get_sequences(path)
    for seq_path in seq_paths:
        pcd_dir = os.path.join(seq_path, 'ars548', 'points')
        if not os.path.exists(pcd_dir):
            print(f"Expected .pcd files in {pcd_dir}, but folder not found.")
            continue
        for fname in sorted(os.listdir(pcd_dir)):
            if not fname.endswith(".pcd"):
                continue
            pc_path = os.path.join(pcd_dir, fname)
            pcls.append(pc_path)
    return pcls

def get_pcls_from_paths(paths):
    pcls = []
    for path in paths:
        pcls.extend(get_pcls_from_path(path))
    return pcls

def get_max_pcd_size_in_data(pcls):
    """
    Returns the maximum number of points in a .pcd from the provided list of .pcd files.
    Used for padding and keeping the input length uniform.
    
    Args:
        pcls: list of .pcd file paths

    Returns:
        int: maximum number of points in any PCD file
    """
    max_size = 0
    for pc_path in pcls:
        try:
            pc = PointCloud.from_path(pc_path)
            n_points = pc.points
            if n_points > max_size:
                max_size = n_points
        except Exception as e:
            print(f"[Warning] : Failed to read {pc_path}: {e}")
            continue
    return max_size

def read_snail_gt_odometry(sequence_path):
    odom_path = os.path.join(sequence_path, "gt_odometry.txt")
    if not os.path.exists(odom_path):
        raise FileNotFoundError(f"No gt_odometry.txt found at {odom_path}")

    gt_time = []
    gt_position = []
    gt_orientation = []

    with open(odom_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue

            timestamp_str = parts[0]
            ts = int(timestamp_str.split('.')[0]) * 10**9 + int((timestamp_str.split('.')[1] + '000000000')[:9])
            x, y, z, qx, qy, qz, qw = map(float, parts[1:])
            gt_time.append(ts)
            gt_position.append([x, y, z])
            gt_orientation.append([qx, qy, qz, qw])

    gt_time = np.asarray(gt_time, dtype=np.int64)[:, np.newaxis]
    gt_position = np.asarray(gt_position, dtype=np.float32)
    gt_orientation = np.asarray(gt_orientation, dtype=np.float32)
    return gt_time, gt_position, gt_orientation

def remove_dc_noise(points_xyzvi, threshold_x=0.1, threshold_y=0.1):
    cond_x_1 = points_xyzvi[:, 0] < threshold_x
    cond_y_1 = points_xyzvi[:, 1] < threshold_y
    cond_y_2 = points_xyzvi[:, 1] > -threshold_y
    joint_cond = cond_x_1 & (cond_y_1 & cond_y_2)
    points_xyzvi = np.delete(points_xyzvi, np.where(joint_cond), axis=0)
    return points_xyzvi

def apply_radar_to_xt32_transform(pointclouds_as_mat_list, T_xt32_radar):
    pointclouds_transformed = []
    rotation = T_xt32_radar[:3, :3]
    translation = T_xt32_radar[:3, 3]
    for pointcloud in pointclouds_as_mat_list:
        pc_t = pointcloud.copy()
        xyz = pc_t[:, :3]
        xyz_t = (rotation @ xyz.T + translation[:, None]).T
        pc_t[:, :3] = xyz_t
        pointclouds_transformed.append(pc_t)
    return pointclouds_transformed

def build_input_to_model(previous_pointcloud, current_pointcloud, max_input_size):
    first_pointcloud = remove_dc_noise(
        previous_pointcloud)
    second_pointcloud = remove_dc_noise(
        current_pointcloud)
    needed_zeros_first_pointcloud = max_input_size - \
        first_pointcloud.shape[0] if max_input_size - \
        first_pointcloud.shape[0] > 0 else 0
    first_pointcloud_padded = np.pad(
        first_pointcloud[:max_input_size, :], ((1, needed_zeros_first_pointcloud), (0, 0)), "constant", constant_values=(0, 0))
    # 3. Pad with zeros after transformation.
    needed_zeros_second_pointcloud = max_input_size - \
        second_pointcloud.shape[0] if max_input_size - \
        second_pointcloud.shape[0] > 0 else 0
    second_pointcloud_padded = np.pad(
        second_pointcloud[:max_input_size, :], ((1, needed_zeros_second_pointcloud), (0, 0)), "constant", constant_values=(0, 0))
    # 4. Concatenate both pointclouds as one input entry.
    input_data_block = np.hstack((first_pointcloud_padded.reshape((1, first_pointcloud_padded.size)),
                                 second_pointcloud_padded.reshape((1, second_pointcloud_padded.size))))
    return input_data_block

class DataFactory:
    """write pc2 and gt for training from all the bags into files
    inside `data` folder. Write as `.hdf5` files. Files are:
    `pointclouds.hdf5` and `labels.hdf5`.
    """

    def __enter__(self):
        # Remove old files.
        if self.mode == "train":
            examples_path = "./extended_data_features/train/pointclouds.hdf5"
            labels_path = "./extended_data_features/train/labels.hdf5"
            timestamps_path = "./extended_data_features/train/timestamps.hdf5"
        elif self.mode == "val":
            examples_path = "./extended_data_features/val/pointclouds.hdf5"
            labels_path = "./extended_data_features/val/labels.hdf5"
            timestamps_path = "./extended_data_features/val/timestamps.hdf5"
        elif self.mode == "test":
            examples_path = "./extended_data_features/test/pointclouds.hdf5"
            labels_path = "./extended_data_features/test/labels.hdf5"
            timestamps_path = "./extended_data_features/test/timestamps.hdf5"
        else:
            raise ValueError("Unknown mode.")

        Path(examples_path).unlink(missing_ok=True)
        Path(labels_path).unlink(missing_ok=True)

        # Training files.
        self.pointclouds_file = h5py.File(examples_path, 'a')
        self.labels_file = h5py.File(labels_path, 'a')
        self.timestamps_file = h5py.File(timestamps_path, 'a')
        n_pointclouds_in_input = 2
        n_coords_in_point = 5
        # Add 1 to make the non-matched class.
        self.examples_dset = self.pointclouds_file.create_dataset(
            "examples", shape=(1, n_pointclouds_in_input * n_coords_in_point * self.pc2_max_size_in_data + 10),
            chunks=True, maxshape=(None, n_pointclouds_in_input * n_coords_in_point * self.pc2_max_size_in_data + 10))
        self.labels_dset = self.labels_file.create_dataset(
            "labels", shape=(1, n_coords_in_point * self.pc2_max_size_in_data),
            chunks=True, maxshape=(None, n_coords_in_point * self.pc2_max_size_in_data))
        self.timestamps_dset = self.timestamps_file.create_dataset(
            "timestamps", shape=(0, 2), maxshape=(None, 2), chunks=True, dtype=np.int64
        )
        # Below only relevant when normalizing the data.
        if self.mode == "train":
            # Normalization file-mapped arrays.
            self.pointclouds_norm_file = np.memmap(
                self.pointclouds_norm_path, dtype='float32', mode='w+', shape=(1, 5))
            self.labels_norm_file = np.memmap(
                self.labels_norm_path, dtype='float32', mode='w+', shape=(1, 3))
        return self

    def add_examples_and_labels_for_normalization(self, examples, labels):
        labels = np.diff(labels, axis=0)
        # debug
        self.gt_debug_non_interp = np.vstack(
            (self.gt_debug_non_interp, labels))
        # debug
        # Remove DC-corrupted values.
        examples = np.vstack(examples)
        examples = remove_dc_noise(examples)
        self.pointclouds_norm_file = np.vstack(
            (self.pointclouds_norm_file, examples))
        self.labels_norm_file = np.vstack((self.labels_norm_file, labels))

    def DEBUG_flush_gt_buffer(self):
        # debug
        self.gt_debug = np.empty(shape=(0, 3 + 4), dtype=np.float64)
        self.gt_debug_non_interp = np.empty(shape=(0, 3), dtype=np.float64)
        self.gt_debug_time = np.empty(shape=(0, 1), dtype=np.float64)
        # debug

    def calculate_normalization_params(self, normalizer):
        # Remove first rows which have zeros.
        self.pointclouds_norm_file = np.delete(
            self.pointclouds_norm_file, (0), axis=0)
        self.labels_norm_file = np.delete(
            self.labels_norm_file, (0), axis=0)
        normalizer.calculate_normalization_params(
            self.pointclouds_norm_file, self.labels_norm_file)

    def __exit__(self, exc_type, exc_value, traceback):
        self.pointclouds_file.close()
        self.labels_file.close()
        self.timestamps_file.close()
        if self.mode == "train":
            print(
                "Removing file-mapped arrays used for calculating normalization params ... .")
            Path(self.pointclouds_norm_path).unlink(missing_ok=True)
            Path(self.labels_norm_path).unlink(missing_ok=True)

    def __init__(self, pc2_max_size_in_data, mode):
        # debug
        self.gt_debug = np.empty(shape=(0, 3 + 4), dtype=np.float64)
        self.gt_debug_non_interp = np.empty(shape=(0, 3), dtype=np.float64)
        self.gt_debug_time = np.empty(shape=(0, 1), dtype=np.float64)
        # debug
        self.mode = mode
        self.pc2_max_size_in_data = pc2_max_size_in_data
        self.row_number = 0
        if self.mode == "train":
            # Normalization files.
            self.pointclouds_norm_path = "./extended_data_features/norm/pointclouds.memmap"
            self.labels_norm_path = "./extended_data_features/norm/labels.memmap"

    def _write_to_hdf5(self, gt_data_block, input_data_block, t1, t2):
        """Here we pad the data to the maximum length and write into hdf5.
        """
        self.row_number = self.row_number + 1
        self.labels_dset.resize(
            self.row_number, axis=0)
        self.labels_dset[(self.row_number - 1):] = gt_data_block
        self.examples_dset.resize(
            self.row_number, axis=0)
        self.examples_dset[(self.row_number - 1):] = input_data_block
        self.timestamps_dset.resize(
            self.row_number, axis=0)
        self.timestamps_dset[(self.row_number - 1):] = np.array([[t1, t2]], dtype=np.int64)

    def _find_closest(self, A, target):
        # A must be sorted.
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A)-1)
        left = A[idx - 1]
        right = A[idx]
        idx -= target - left < right - target
        return idx

    def generate_gt_and_input_data(self, gt_time, gt_position, gt_orientation,
                               radar_time, pointclouds_as_xyzvi_mat_list):
        """Generate radar training pairs with GT poses interpolated to radar timestamps."""
        # Convert GT poses into SE3 matrices
        T_list = []
        for pos, quat in zip(gt_position, gt_orientation):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            T_list.append(T)
        gt_time = gt_time.squeeze()

        # Filter out radar timestamps outside GT range
        valid_mask = (radar_time >= gt_time[0]) & (radar_time <= gt_time[-1])
        radar_time = radar_time[valid_mask]
        pointclouds_as_xyzvi_mat_list = [
            p for p, m in zip(pointclouds_as_xyzvi_mat_list, valid_mask) if m
        ]

        if len(radar_time) < 2:
            print("[WARN] Skipping sequence: not enough valid radar scans.")
            return

        # Prepare SLERP object once
        slerp_obj = Slerp(gt_time, R.from_quat(gt_orientation))

        # Interpolate GT pose for each radar timestamp
        T_interp = []
        for ts in radar_time:
            if ts <= gt_time[0]:
                T_interp.append(T_list[0])
                continue
            if ts >= gt_time[-1]:
                T_interp.append(T_list[-1])
                continue
            i = np.searchsorted(gt_time, ts) - 1
            t0, t1 = gt_time[i], gt_time[i + 1]
            a = (ts - t0) / (t1 - t0)
            p0, p1 = gt_position[i], gt_position[i + 1]
            R_interp = slerp_obj([ts])[0].as_matrix()
            T = np.eye(4)
            T[:3, :3] = R_interp
            T[:3, 3] = (1 - a) * p0 + a * p1
            T_interp.append(T)

        for idx in range(len(radar_time) - 1):
            T_rel = np.linalg.inv(T_interp[idx]) @ T_interp[idx + 1]
            rel_rot = T_rel[:3, :3]
            rel_pos = T_rel[:3, 3][np.newaxis, :]

            input_data_block = build_input_to_model(
                pointclouds_as_xyzvi_mat_list[idx],
                pointclouds_as_xyzvi_mat_list[idx + 1],
                self.pc2_max_size_in_data
            )

            first_pointcloud = remove_dc_noise(pointclouds_as_xyzvi_mat_list[idx])
            needed_zeros_first_pointcloud = self.pc2_max_size_in_data - first_pointcloud.shape[0]
            xyz = first_pointcloud[:, :3]
            di = first_pointcloud[:, 3:] 
            xyz_transformed = (
                rel_rot.T @ (xyz - rel_pos).T
            ).T
            transformed_first_pointcloud = np.hstack((xyz_transformed, di))
            '''transformed_first_pointcloud = (
                rel_rot.T @ (first_pointcloud - rel_pos).T
            ).T'''
            transformed_first_pointcloud = np.pad(
                transformed_first_pointcloud,
                ((0, needed_zeros_first_pointcloud), (0, 0)),
                "constant",
                constant_values=(0, 0),
            )
            gt_data_block = transformed_first_pointcloud.reshape(
                (1, transformed_first_pointcloud.size)
            )

            t1 = radar_time[idx]
            t2 = radar_time[idx + 1]
            Δt = (t2 - t1) * 1e-9
            print(f"Pair {idx}: Δt={Δt:.4f}s")

            # Write to hdf5
            self._write_to_hdf5(gt_data_block, input_data_block, t1, t2)
