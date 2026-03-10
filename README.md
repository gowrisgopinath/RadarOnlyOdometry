# Radar transformer
This repo includes base changes from [Radar transformer](https://github.com/aau-cns/radar_transformer/tree/main) and adapted to train and evaluate on sparse Snail 4D radar dataset using STEAM library

# Build STEAM library
```bash
cd cpp
mkdir -p build
cd build
cmake ..
make
```

# Prepare dataset for training

1. Dataset preparation
```bash
python3 prepare_dataset.py
```
This creates dataset in hdf5 format with below formats:
pointclouds.hdf5 - flattened consecutive radar frame pairs [pc1, pc2] transformed to lidar frame (to align with gt poses)
labels.hdf5 - pc1 transformed to pc2 frame using gt pose
timestamps.hdf5 - radar timestamp pairs [t1, t2]

2. Dataset verification

The accuracy of the dataset prepared above can be verified by passing pc1 (from pointclouds.hdf5)  and gt transformed pc1 (labels.hdf5) as correspondences to STEAM backend. STEAM then estimates the trajectory based on Gauss Newton optimization on the measurement factors generated from these correspondences.
```bash
python3 verify_snail_gt_steam.py
evo_traj tum steam_pred_trajectory.txt -p
```
# Train radar transformer
To train the network in a supervised manner, run below command
```bash
python3 main.py
```

# Evaluate learned correspondences using STEAM
To evaluate the pre-trained network using STEAM, run below command
```bash
python3 evaluate.py
```
Set NUM_FEATURES_PER_POINT = 3 or 4 or 5 depending on the number of features (x, y, z, doppler, rcs) used for training radar transformer.

Use below commands to get the translation and rotational relative pose errors
```bash
evo_rpe tum steam_gt_pred_trajectory.txt steam_pred_trajectory_xyz.txt --align --correct_scale --delta 1 --delta_unit m -r full --save_results results/rpe_full_xyz.zip --save_plot results/rpe_full_xyz.pdf --plot_mode xy

evo_rpe tum steam_gt_pred_trajectory.txt steam_pred_trajectory_xyz.txt --align --correct_scale --delta 1 --delta_unit m -r trans_part --save_results results/rpe_trans_xyz.zip --save_plot results/rpe_trans_xyz.pdf --plot_mode xy

evo_rpe tum steam_gt_pred_trajectory.txt steam_pred_trajectory_xyz.txt --align --correct_scale --delta 1 --delta_unit m -r rot_part --save_results results/rpe_rot_xyz.zip --save_plot results/rpe_rot_xyz.pdf --plot_mode xy
```

# Base paper citation
```
@misc{michalczyk2025learningpointcorrespondencesradar,
      title={Learning Point Correspondences In Radar 3D Point Clouds For Radar-Inertial Odometry}, 
      author={Jan Michalczyk and Stephan Weiss and Jan Steinbrener},
      year={2025},
      eprint={2506.18580},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.18580}, 
}
```
