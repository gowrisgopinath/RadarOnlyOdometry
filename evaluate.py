import os
import h5py
import torch
import numpy as np
import argparse
from pathlib import Path
import networks.radar_transformer
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
from networks.steam_solver import SteamSolver
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import ConnectionPatch

NUM_FEATURES_PER_POINT = 5

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
            ts = ts / 1e9  # convert ns to s
            t = T[:3, 3]
            q = R.from_matrix(T[:3, :3]).as_quat()
            f.write(
                f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
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
    pc1_size = np.max(np.count_nonzero(X[1:half_input, :], axis=0), axis=0)
    pc2_size = np.max(np.count_nonzero(X[(half_input + 1):, :], axis=0), axis=0)
    return int(pc1_size), int(pc2_size)

def evaluate(model, pcs, gts, timestamps, solver, device):
    rel_transforms = []
    timestamps_abs = []

    num_pairs = pcs.shape[0]
    print(f"\nEvaluating {num_pairs} radar frame pairs...\n")

    model.eval()
    with torch.no_grad():
        for idx in range(num_pairs):
            pc_flat = pcs[idx]
            print(pc_flat.shape)
            gt_flat = gts[idx]
            print(gt_flat.shape)
            t1, t2 = timestamps[idx]
            print(f't1 : {t1}, t2 : {t2}')
            delta_t = t2 - t1

            total_len = pc_flat.shape[0]
            half_len = total_len // 2
            if False:
                pc1 = pc_flat[:half_len].reshape(-1, NUM_FEATURES_PER_POINT)
                pc1_size = np.max(np.count_nonzero(pc1[1:, :], axis=0), axis=0)
                print(f'pc1_size : {pc1_size}')
                pc1_gt = gt_flat.reshape(-1, NUM_FEATURES_PER_POINT)
                pc1_gt_size = np.max(np.count_nonzero(pc1_gt, axis=0), axis=0)
                print(f'pc1_gt_size : {pc1_gt_size}')

            pc1 = pc_flat[:half_len].reshape(-1, NUM_FEATURES_PER_POINT)
            pc2 = pc_flat[half_len:].reshape(-1, NUM_FEATURES_PER_POINT)
            pc1_gt = gt_flat.reshape(-1, NUM_FEATURES_PER_POINT)
            print("pc1.shape before flatten:", pc1.shape)
            print("pc2.shape before flatten:", pc2.shape)
            print("pc1_gt.shape before padding:", pc1_gt.shape)

            pc1_gt_padded = np.pad(
                pc1_gt,
                ((1, 0), (0, 0)),
                mode="constant",
                constant_values=0
            )
            print("pc1_gt_padded.shape after padding:", pc1_gt_padded.shape)

            pc1_flat = pc1.flatten()
            pc2_flat = pc2.flatten()
            pc1_gt_flat = pc1_gt_padded.flatten()
            input_concat = np.hstack((pc1_flat, pc2_flat)).reshape(1, -1)

            print("pc1_flat.shape:", pc1_flat.shape)
            print("pc2_flat.shape:", pc2_flat.shape)
            print("input_concat.shape:", input_concat.shape)

            X = torch.from_numpy(input_concat).to(device)
            pred = model(X)
            print("pred.shape:", pred.shape)
            pred_np = pred.squeeze().detach().cpu().numpy()

            plt.figure(figsize=(6, 5))
            plt.imshow(pred_np, cmap="jet", aspect="auto")
            plt.colorbar(label="Matching Confidence")
            plt.title(f"Radar Transformer Affinity Map (Frame {idx})")
            plt.xlabel("Points in point cloud 2")
            plt.ylabel("Points in point cloud 1")
            plt.tight_layout()
            #plt.show()
            #plt.savefig(f"plots/rawaffinity_frame_{idx}.png")
            plt.close()
            pc1_size = np.max(np.count_nonzero(pc1[1:, :], axis=0), axis=0)
            print(f'pc1_size : {pc1_size}')
            pc2_size = np.max(np.count_nonzero(pc2[1:, :], axis=0), axis=0)
            print(f'pc2_size : {pc2_size}')

            pred = pred.squeeze(0)          # shape: (N1, N2) or (N2, N1)
            pred = pred.T  
            affinity = pred[1:pc1_size, 1:pc2_size]
            affinity = torch.softmax(affinity, dim=-1)  # (pc1_size, pc2_size).squeeze(0)
            print(f'affinity shape: {affinity.shape}')
            affinity_np = affinity.detach().cpu().numpy()
            plt.figure(figsize=(6, 5))
            plt.imshow(affinity_np, cmap="jet", aspect="auto")
            plt.colorbar(label="Affinity matrix")
            plt.title(f"Radar Transformer Affinity Map (Frame {idx})")
            plt.xlabel("Points in point cloud 2")
            plt.ylabel("Points in point cloud 1")
            plt.tight_layout()
            #plt.show()
            #plt.savefig(f"plots/affinity_frame_{idx}.png")
            plt.close()

            row_ind, col_ind = linear_sum_assignment(affinity_np, maximize=True)
            idx_pc1 = torch.as_tensor(row_ind, device=device)
            idx_pc2 = torch.as_tensor(col_ind, device=device)
            match_weights = affinity[idx_pc1, idx_pc2]

            top_n = min(180, match_weights.numel())
            if match_weights.numel() > top_n:
                topk_vals, topk_idx = torch.topk(match_weights, k=top_n)
                idx_pc1 = idx_pc1[topk_idx]
                idx_pc2 = idx_pc2[topk_idx]
                match_weights = topk_vals

            if idx_pc1.numel() < 3:
                print(f"[WARN] Frame {idx}: too few matches, skipping.")
                R_pred = torch.eye(3, device=device).unsqueeze(0)
                t_pred = torch.zeros(3, 1, device=device).unsqueeze(0)
                rel_transforms.append({"R": R_pred, "t": t_pred})
                timestamps_abs.append(float(t2))
                continue       

            keypoint_coords = torch.from_numpy(pc1[idx_pc1.cpu().numpy()+1]).float().to(device)
            pseudo_coords = torch.from_numpy(pc2[idx_pc2.cpu().numpy()+1]).float().to(device)
            weights = match_weights.unsqueeze(1)
            print('keypoint_coords', keypoint_coords)
            print('pseudo_coords', pseudo_coords)
            print('weights', weights)

            if idx % 50 == 0:
                fig = plt.figure(figsize=(7,6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pc1[:,0], pc1[:,1], pc1[:,2], c='b', s=2, label='pc1')
                ax.scatter(pc2[:,0], pc2[:,1], pc2[:,2], c='r', s=2, label='pc2')
                
                for i1, i2 in zip(idx_pc1.cpu().numpy(), idx_pc2.cpu().numpy()):
                    
                    p1, p2 = pc1[i1], pc2[i2]
                    print('inside', p1, p2)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'g-', linewidth=0.3)
                
                ax.legend()
                plt.title(f"Frame {idx}: Matched Keypoints")
                #plt.savefig(f"plots/matches_frame_{idx:03d}.png")
                #plt.show()
                plt.close()

            try:
                R_pred, t_pred = solver.optimize(
                    keypoint_coords.unsqueeze(0),
                    pseudo_coords.unsqueeze(0),
                    weights.unsqueeze(0),
                    time_tgt=t2, time_src=t1,
                    t_ref_tgt=t2, t_ref_src=t1
                )
            except Exception as e:
                print(f"[ERR] STEAM failed at frame {idx}: {e}")
                R_pred = torch.eye(3, device=device).unsqueeze(0)
                t_pred = torch.zeros(3, 1, device=device).unsqueeze(0)

            if False:
                if idx == 0:
                    print(f"pc1 shape: {pc1.shape}, pc1_gt shape: {pc1_gt.shape}")

                keypoints = torch.from_numpy(pc1[1:4]).unsqueeze(0).float().to(device)
                pseudo_coords = torch.from_numpy(pc1_gt[:3]).unsqueeze(0).float().to(device)
                weights = torch.full((1, pc1.shape[0]), 0.5, device=device).float()

                print(f"[{idx+1}/{num_pairs}] t1={t1}, t2={t2}, delta_t ={delta_t} ns")

                try:
                    R_pred, t_pred = solver.optimize(
                        keypoints, pseudo_coords, weights,
                        time_tgt=t2, time_src=t1,
                        t_ref_tgt=t2, t_ref_src=t1
                    )
                except Exception as e:
                    print(f"[ERR] STEAM failed at frame {idx}: {e}")
                    R_pred = torch.eye(3, device=device).unsqueeze(0)
                    t_pred = torch.zeros(3, 1, device=device).unsqueeze(0)

            rel_transforms.append({"R": R_pred, "t": t_pred})
            timestamps_abs.append(float(t2))

            print(f"[{idx+1}/{num_pairs}] delta_t={delta_t}ns "
                  f"t_pred={t_pred.squeeze().cpu().numpy()}")

    print("\n Evaluation complete.\n")
    return rel_transforms, timestamps_abs

def main(args):
    model_output_dir = args.model_output_dir or "saved_models_features"
    model_name = args.model_name or "RadarTransformer_epoch25.ptm"
    path_to_model = Path(model_output_dir) / model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "./extended_data_features/test"
    pcs, gts, timestamps = load_hdf5_dataset(data_dir)
    print(f"Loaded dataset: {pcs.shape[0]} samples from {data_dir}")

    input_len = pcs.shape[1]

    model = radar_transformer.RadarDeepMatcher(input_len).to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=device, weights_only=True))
    model.eval()

    config = {
        "gpuid": device,
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

    rel_transforms, timestamps_abs = evaluate(model, pcs, gts, timestamps, solver, device)

    traj = integrate_trajectory(rel_transforms)

    save_path = "steam_pred_trajectory.txt"
    save_trajectory_tum(traj, timestamps_abs, save_path)
    print(f"Trajectory saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_dir", help="Folder where models are stored.")
    parser.add_argument("--model_name", help="Model filename (e.g. RadarTransformer_epoch30.ptm).")
    args = parser.parse_args()
    main(args)
