import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pypcd4 import PointCloud
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from transform import slerp_se3, se2_from_T, R_to_euler_XYZ
from utils import apply_radar_to_xt32_transform

def get_sequences(path):
    """Return list of sequence subfolders in the given root path."""
    seqs = [os.path.join(path, d) for d in os.listdir(path)]
    seqs = [d for d in seqs if os.path.isdir(d)]
    seqs.sort()
    return seqs


def extract_frames(path):
    """Load all .pcd radar scans in a sequence."""
    frames = []
    pcd_dir = os.path.join(path, 'ars548', 'points')
    if not os.path.exists(pcd_dir):
        print(f"[WARN] No 'ars548/points' folder found in {path}")
        return []

    for fname in sorted(os.listdir(pcd_dir)):
        if not fname.endswith(".pcd"):
            continue
        pc_path = os.path.join(pcd_dir, fname)
        pc = PointCloud.from_path(pc_path)
        fields = ['x', 'y', 'z', 'intensity', 'doppler']
        frame_array = np.stack([pc.pc_data[f].astype(np.float32) for f in fields], axis=1)
        base = os.path.splitext(fname)[0]
        sec = int(base.split('.')[0])
        nsec = int((base.split('.')[1] + '000000000')[:9])
        ts = sec * 10**9 + nsec  # nanoseconds
        frames.append((ts, frame_array))
    return frames


def load_T_world_xt32(sequence_path):
    """Load ground-truth odometry (SE3) for the sequence."""
    T_w_xt32 = {}
    odom_path = os.path.join(sequence_path, "gt_odometry.txt")
    if not os.path.exists(odom_path):
        raise FileNotFoundError(f"No gt_odometry.txt found in {sequence_path}")
    with open(odom_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            ts_str = parts[0]
            x, y, z, qx, qy, qz, qw = map(float, parts[1:])
            ts = int(ts_str.split('.')[0]) * 10**9 + int((ts_str.split('.')[1] + '000000000')[:9])
            T = np.eye(4)
            T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [x, y, z]
            T_w_xt32[ts] = T
    return T_w_xt32


def fit_ransac_static_mask(frame_array,
                           degree=2,
                           residual_thresh=0.7,
                           min_samples=60,
                           random_state=0):
    """RANSAC static/dynamic filtering on Doppler vs DoA."""
    if frame_array.size == 0:
        return np.ones((0,), dtype=bool), None
    x, y = frame_array[:, 0], frame_array[:, 1]
    doa = np.arctan2(y, x).reshape(-1, 1)
    doppler = frame_array[:, 4].astype(np.float64)
    n = len(doppler)
    min_samples_eff = max(2, min(min_samples, n // 2))
    model = make_pipeline(
        PolynomialFeatures(degree, include_bias=True),
        RANSACRegressor(
            residual_threshold=residual_thresh,
            min_samples=min_samples_eff,
            random_state=random_state
        )
    )
    model.fit(doa, doppler)
    inlier_mask = model.named_steps['ransacregressor'].inlier_mask_
    if inlier_mask is None:
        inlier_mask = np.ones(n, dtype=bool)
    return inlier_mask, model


def pad_pointcloud_for_transformer(pc, max_points):
    """
    Pads or subsamples radar point clouds to fit RadarTransformer input:
      - Row 0: all zeros (for 'no match' class)
      - Then: real points
      - Then: trailing zeros to reach (max_points + 1) total rows
    """
    n = pc.shape[0]
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        pc = pc[idx]
    needed_zeros_after = max_points - pc.shape[0]
    pc_padded = np.pad(pc, ((1, needed_zeros_after), (0, 0)),
                       mode='constant', constant_values=0)
    return pc_padded


def rpy_from_R(Rm):
    """Convert rotation matrix to roll, pitch, yaw (radians)."""
    return R.from_matrix(Rm).as_euler('xyz', degrees=False)

class SnailRadarDataset(Dataset):
    def __init__(self,
                 sequence_path,
                 use_ransac_static=False,
                 ransac_degree=2,
                 ransac_residual=0.7,
                 ransac_min_samples=60):

        self.frames = extract_frames(sequence_path)
        self.frames.sort(key=lambda pair: pair[0])
        if len(self.frames) < 2:
            raise ValueError(f"Not enough frames in {sequence_path}")

        start_ts_ns = self.frames[0][0]
        self.frames = [(ts - start_ts_ns, f) for ts, f in self.frames]  # relative in ns
        print(f"[{os.path.basename(sequence_path)}] Radar time normalized to start=0.0 s")

        self.max_points = max(f.shape[0] for _, f in self.frames)
        print(f"[{os.path.basename(sequence_path)}] Max points per frame: {self.max_points}")

        self.T_w_xt32 = load_T_world_xt32(sequence_path)
        self.gt_ts_sorted = np.array(sorted(self.T_w_xt32.keys()), dtype=np.int64)
        self.gt_T_list = [self.T_w_xt32[t] for t in self.gt_ts_sorted]

        gt_start_ns = self.gt_ts_sorted[0]
        self.gt_ts_sorted = (self.gt_ts_sorted - gt_start_ns)

        min_ts, max_ts = self.gt_ts_sorted[0], self.gt_ts_sorted[-1]
        self.frames = [(ts, f) for ts, f in self.frames if min_ts <= ts <= max_ts]
        print(f"[{os.path.basename(sequence_path)}] Frames retained: {len(self.frames)}")

        self.use_ransac_static = use_ransac_static
        if self.use_ransac_static:
            self.frames = self._apply_ransac_static(self.frames, ransac_degree, ransac_residual, ransac_min_samples)

        self.T_xt32_r = np.array([
            [-0.0148246946,  0.999869705,  -0.006387675, 0],
            [-0.967459493,   -0.012729749,  0.252705526,  0],
            [ 0.252591285,    0.009926099,   0.967522152,  0.07],
            [ 0,              0,              0,            1]
        ])

    def _apply_ransac_static(self, frames, degree, residual, min_samples):
        out_frames = []
        for ts, fa in frames:
            if fa.shape[0] < 5:
                out_frames.append((ts, fa))
                continue
            mask, _ = fit_ransac_static_mask(
                fa, degree=degree,
                residual_thresh=residual,
                min_samples=min_samples
            )
            out_frames.append((ts, fa[mask] if mask.any() else fa))
        print(f"RANSAC filtering done. Avg pts/frame: {np.mean([len(f) for _, f in out_frames]):.1f}")
        return out_frames

    def __len__(self):
        return len(self.frames) - 1

    def rel_pose_vec6(self, t_a, t_b):
        """Compute SE3 relative transform (6D vector) between two timestamps (seconds)."""
        T_w_xt32_a = slerp_se3(t_a, self.gt_ts_sorted, self.gt_T_list)
        T_w_xt32_b = slerp_se3(t_b, self.gt_ts_sorted, self.gt_T_list)
        T_12 = np.linalg.inv(T_w_xt32_a) @ T_w_xt32_b
        t = T_12[:3, 3]
        rpy = rpy_from_R(T_12[:3, :3])
        return np.concatenate([t, rpy], axis=0)

    def __getitem__(self, idx):
        ts_prev, f_prev = self.frames[idx]
        ts_curr, f_curr = self.frames[idx + 1]

        f_prev_lidar, f_curr_lidar = apply_radar_to_xt32_transform([f_prev[:, :3], f_curr[:, :3]], self.T_xt32_r)

        pc_prev = pad_pointcloud_for_transformer(f_prev_lidar, self.max_points)
        pc_curr = pad_pointcloud_for_transformer(f_curr_lidar, self.max_points)

        rel_vec6 = self.rel_pose_vec6(ts_prev, ts_curr)
        timestamps = [ts_prev, ts_curr]
        t_ref = timestamps.copy()

        return {
            'pc_prev': torch.tensor(pc_prev, dtype=torch.float32),
            'pc_curr': torch.tensor(pc_curr, dtype=torch.float32),
            'timestamps': timestamps,
            't_ref': t_ref,
            'T_12': torch.tensor(rel_vec6, dtype=torch.float32)
        }

def get_dataloaders(sequence_path, config):
    print(f"Using single sequence: {sequence_path}")

    # Common dataset arguments
    common_kwargs = dict(
        use_ransac_static=config.get('use_ransac_static', False),
        ransac_degree=config.get('ransac_degree', 2),
        ransac_residual=config.get('ransac_residual', 0.7),
        ransac_min_samples=config.get('ransac_min_samples', 60)
    )

    dataset = SnailRadarDataset(sequence_path, **common_kwargs)

    return DataLoader(
            dataset,
            batch_size=config.get('batch_size', 1),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2
        )
