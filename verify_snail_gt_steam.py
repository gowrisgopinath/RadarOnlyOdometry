import os
import h5py
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from networks.steam_solver import SteamSolver

def integrate_trajectory(relative_transforms):
    trajectory = []
    
    T_curr = np.eye(4, dtype=np.float64)
    trajectory.append(T_curr.copy())

    for Rt in relative_transforms:
        R_pred = Rt["R"].detach().cpu().numpy()
        t_pred = Rt["t"].detach().cpu().numpy().reshape(-1, 1)

        if R_pred.ndim == 3:
            R_pred = R_pred[0]
        if t_pred.ndim == 3:
            t_pred = t_pred[0]

        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = R_pred
        T_rel[:3, 3:] = t_pred

        T_curr = T_curr @ T_rel
        trajectory.append(T_curr.copy())

    return np.array(trajectory)  # (N, 4, 4)

def save_trajectory_tum(traj_np, timestamps, save_path):
    with open(save_path, "w") as f:
        for ts, T in zip(timestamps, traj_np):
            ts = ts / 1e9  # convert ns -> s
            t = T[:3, 3]
            q = R.from_matrix(T[:3, :3]).as_quat()
            f.write(
                f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            )

def integrate_trajectory_se2(relative_transforms):
    trajectory = []
    current_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    trajectory.append(current_pose.copy())

    for Rt in relative_transforms:
        R_pred = Rt["R"].cpu().numpy()
        t_pred = Rt["t"].cpu().numpy().flatten()

        if R_pred.ndim == 3:
            R_pred = R_pred[0]

        dx, dy = t_pred[0], t_pred[1]
        dtheta = np.arctan2(R_pred[1, 0], R_pred[0, 0])

        cos_t, sin_t = np.cos(current_pose[2]), np.sin(current_pose[2])
        x_new = current_pose[0] + cos_t * dx - sin_t * dy
        y_new = current_pose[1] + sin_t * dx + cos_t * dy
        theta_new = np.arctan2(np.sin(current_pose[2] + dtheta),
                               np.cos(current_pose[2] + dtheta))

        current_pose = np.array([x_new, y_new, theta_new], dtype=np.float32)
        trajectory.append(current_pose.copy())

    return np.array(trajectory)

def save_trajectory_tum_se2(traj_np, timestamps, save_path):
    with open(save_path, "w") as f:
        for (ts, (x, y, yaw)) in zip(timestamps, traj_np):
            ts = ts / 1e9
            quat = R.from_euler("z", yaw).as_quat()
            f.write(
                f"{ts:.6f} {x:.6f} {y:.6f} 0.000000 "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
            )

def load_hdf5_dataset(data_dir):
    with h5py.File(os.path.join(data_dir, "pointclouds.hdf5"), "r") as f_pc, \
         h5py.File(os.path.join(data_dir, "labels.hdf5"), "r") as f_gt, \
         h5py.File(os.path.join(data_dir, "timestamps.hdf5"), "r") as f_ts:

        pcs = f_pc["examples"][:]
        gts = f_gt["labels"][:]
        timestamps = f_ts["timestamps"][:]
    return pcs, gts, timestamps

def calculate_non_padded_pc_sizes(X, half_input):
    pc1_size = np.max(np.count_nonzero(
        X[1:half_input, :], axis=0), axis=0)
    pc2_size = np.max(np.count_nonzero(
        X[(half_input + 1):, :], axis=0), axis=0)
    return pc1_size, pc2_size

def main():
    data_dir = "./extended_data_features/test"
    gpu = "cuda" if torch.cuda.is_available() else "cpu"

    pcs, gts, timestamps = load_hdf5_dataset(data_dir)
    num_pairs = pcs.shape[0]
    print(f"Loaded {num_pairs} pairs from {data_dir}")

    config = {
        "gpuid": gpu,
        "steam": {
            "time_step": 0.05,
            "qc_diag": [1.0, 1.0, 1.0, 0.1, 0.1, 0.1],
            "use_ctsteam": False,
            "use_ransac": False,
            "ransac_version": 1,
            "expect_approx_opt": 0,
            "ex_rotation_sv": [1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0],
            "ex_translation_vs_in_s": [0.0, 0.0, 0.0],
            "zero_vel_prior": False,
            "vel_prior": False,
        }
    }

    solver = SteamSolver(config)
    rel_transforms = []
    timestamps_abs = [timestamps[0, 0]]

    for idx in range(num_pairs):
        pc_flat = pcs[idx]
        gt_flat = gts[idx]
        t1, t2 = timestamps[idx]
        print(f't1 : {t1}, t2 : {t2}')
        delta_t = t2 - t1

        total_len = pc_flat.shape[0]
        half_len = total_len // 2
        pc1 = pc_flat[:half_len].reshape(-1, 5)
        pc1_size = np.max(np.count_nonzero(pc1[1:, :], axis=0), axis=0)
        print(f'pc1_size : {pc1_size}')
        pc1_gt = gt_flat.reshape(-1, 5)
        pc1_gt_size = np.max(np.count_nonzero(pc1_gt, axis=0), axis=0)
        print(f'pc1_gt_size : {pc1_gt_size}')

        #keypoints = torch.from_numpy(pc1[1:pc1_size+1]).unsqueeze(0).float().to(gpu)
        keypoints = torch.from_numpy(pc1[1:4]).unsqueeze(0).float().to(gpu)
        print('keypoints', keypoints)
        #pseudo_coords = torch.from_numpy(pc1_gt[:pc1_gt_size]).unsqueeze(0).float().to(gpu)
        pseudo_coords = torch.from_numpy(pc1_gt[:3]).unsqueeze(0).float().to(gpu)
        print('pseudo_coords', pseudo_coords)
        weights = torch.ones((1, pc1.shape[0]), device=gpu).float()

        try:
            R_pred, t_pred = solver.optimize(
                keypoints, pseudo_coords, weights,
                time_tgt=t2, time_src=t1,
                t_ref_tgt=t2, t_ref_src=t1
            )
        except Exception as e:
            print(f"[ERR] STEAM failed at pair {idx}: {e}")
            R_pred = torch.eye(3, device=gpu).unsqueeze(0)
            t_pred = torch.zeros(3, 1, device=gpu).unsqueeze(0)

        rel_transforms.append({"R": R_pred, "t": t_pred})
        timestamps_abs.append(float(t2))

        print(f"[{idx+1}/{num_pairs}] delta_t={delta_t:.4f}s "
              f"t_pred={t_pred.squeeze().cpu().numpy()}")

    traj = integrate_trajectory(rel_transforms)

    save_path = "steam_gt_pred_trajectory.txt"
    save_trajectory_tum(traj, timestamps_abs, save_path)
    print(f"Trajectory saved to {save_path}")

if __name__ == "__main__":
    main()
