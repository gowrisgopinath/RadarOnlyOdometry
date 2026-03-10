import numpy as np
import utils
import visualization
import copy
from scipy.spatial.transform import Rotation as R
import os
import glob
from pathlib import Path
from pypcd4 import PointCloud

os.makedirs("./data/data_xyzdi/train", exist_ok=True)
os.makedirs("./data/data_xyzdi/val", exist_ok=True)
os.makedirs("./data/data_xyzdi/test", exist_ok=True)
os.makedirs("./data/data_xyzdi/norm", exist_ok=True)

T_xt32_r = np.array([
        [-0.0148246945939438,  0.999869704931098,  -0.00638767491813221, 0],
        [-0.967459492843472,   -0.0127297492912655,  0.252705525895304,   0],
        [ 0.252591285490489,    0.00992609917652017,  0.967522151941183,   0.07],
        [ 0,                   0,                    0,                  1]
        ])

def main():
    print("Outputting data for train/val/test into hdf5 ... .")

    mode_and_folders = {
        "train": "/workspace/radar/snail/train/", 
        "val": "/workspace/radar/snail/val/",
        "test": "/workspace/radar/snail/test/"}
    mode_and_seqs = {}
    for mode, folder in mode_and_folders.items():
        mode_and_seqs[mode] = utils.get_sequences(folder)

    all_pcls = utils.get_pcls_from_paths(mode_and_folders.values())
    max_pcd_size_in_data = utils.get_max_pcd_size_in_data(all_pcls)
    print('max_pcd_size_in_data : ', max_pcd_size_in_data)
    doppler_stats, rcs_stats = utils.get_doppler_rcs_minmax(all_pcls)
    for mode, seqs in mode_and_seqs.items():
        with utils.DataFactory(max_pcd_size_in_data, mode) as data_factory:
            for seq in seqs:
                print(f"Processing sequence: {seq}")
                gt_time, gt_position, gt_orientation = utils.read_snail_gt_odometry(seq)
                pcd_dir = os.path.join(seq, "ars548", "points")
                pcd_paths = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
                if len(pcd_paths) == 0:
                    print(f"No .pcd files found in {pcd_dir}, skipping.")
                    continue

                pointclouds_as_xyzvi_mat_list = []
                radar_time = []

                for p in pcd_paths:
                    try:
                        pc = PointCloud.from_path(p)
                        doppler = pc.pc_data['doppler']
                        intensity = pc.pc_data['intensity']
                        doppler_norm = (doppler - doppler_stats['min']) / (doppler_stats['max'] - doppler_stats['min'] + 1e-6)
                        rcs_norm = (intensity - rcs_stats['min']) / (rcs_stats['max'] - rcs_stats['min'] + 1e-6)

                        arr = np.stack([
                            pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z'],
                            doppler_norm, rcs_norm
                        ], axis=1)
                        pointclouds_as_xyzvi_mat_list.append(arr)
                        base = os.path.splitext(os.path.basename(p))[0]
                        sec = int(base.split('.')[0]); nsec = int((base.split('.')[1] + '000000000')[:9])
                        ts = sec * 10**9 + nsec
                        radar_time.append(ts)

                    except Exception as e:
                        print(f"Warning: Failed to read {p}: {e}")

                radar_time = np.asarray(radar_time, dtype=np.int64)[:, np.newaxis]  
                # Normalize radar and gt time/position axis.

                radar_time = np.squeeze((radar_time - radar_time[0]))
                print(radar_time)
                print('radar time duration : ', radar_time[-1] - radar_time[0])
                gt_time = np.squeeze((gt_time - gt_time[0]))
                print('gt time duration : ', gt_time[-1] - gt_time[0])
                # Bring poses to the origin.
                gt_position = gt_position - gt_position[0]
                gt_orientation = R.from_matrix(np.matmul(R(gt_orientation[0]).as_matrix(
                ).transpose(), R(gt_orientation).as_matrix())).as_quat()
                
                # Transform radar point clouds to lidar frame to align with gt
                pointclouds_in_xt32_frame_mat_list = utils.apply_radar_to_xt32_transform(pointclouds_as_xyzvi_mat_list, T_xt32_r)
                # Construct the the ground truth and input data and write hdf5.
                data_factory.generate_gt_and_input_data(
                    gt_time, gt_position, gt_orientation, radar_time, pointclouds_in_xt32_frame_mat_list)

if __name__ == '__main__':
    main()
