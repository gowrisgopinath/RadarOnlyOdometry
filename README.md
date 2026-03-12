# Hybrid Radar Odometry pipeline

https://github.com/user-attachments/assets/5948a9d8-3654-4549-965d-cce21685aca4

We are currently exploring the possibility of a radar-only odometry pipeline designed to operate on sparse 4D radar point clouds, which can perform robustly even in adverse weather and low-light conditions.  

This demo video visualizes LiDAR and radar point clouds, the camera view, and odometry results of our pipeline. The camera feed highlights a night time rainy scenario, emphasizing the challenges for vision-based methods. The radar point clouds are sparse and noisy, with roughly 256 points per frame compared to LiDAR’s 10,000 points per frame. The odometry results shown are unoptimized, meaning they do not compensate for accumulated global trajectory drift yet, illustrating the baseline performance of our system.

## Design Pipeline  

This pipeline learns inter-frame correspondences from sparse 4D radar point clouds using a transformer-based front-end for feature extraction and association. The learned correspondences are then passed to the STEAM probabilistic back-end, which performs Gauss–Newton optimization to estimate the vehicle trajectory.  

<img width="1143" height="315" alt="Image" src="https://github.com/user-attachments/assets/a2d74369-7f58-4474-baa2-93487c2a366d" />

This repo includes base ideas and changes from [Radar transformer](https://github.com/aau-cns/radar_transformer/tree/main), [HERO](https://github.com/utiasASRL/hero_radar_odometry), [4DRO-Net](https://ieeexplore.ieee.org/document/10237296), and is adapted to train and evaluate on sparse [SNAIL](https://snail-radar.github.io/) 4D radar dataset using [STEAM](https://github.com/utiasASRL/steam) probabilistic trajectory estimation library

## Usage

### 1. Clone the repo

### 2. Build STEAM library
```bash
cd cpp
mkdir -p build
cd build
cmake ..
make
```

### 3. Prepare dataset for training

1. Dataset preparation
```bash
python3 prepare_dataset.py
```
This creates dataset in hdf5 format with below formats:  
- pointclouds.hdf5 - flattened consecutive radar frame pairs [pc1, pc2] transformed to lidar frame (to align with gt poses)  
- labels.hdf5 - pc1 transformed to pc2 frame using gt pose  
- timestamps.hdf5 - radar timestamp pairs [t1, t2]  

2. Dataset verification

The accuracy of the dataset prepared above can be verified by passing pc1 (from pointclouds.hdf5)  and gt transformed pc1 (labels.hdf5) as correspondences to STEAM backend. STEAM then estimates the trajectory based on Gauss Newton optimization on the measurement factors generated from these correspondences.
```bash
python3 verify_snail_gt_steam.py
evo_traj tum steam_pred_trajectory.txt -p
```
### 4. Train radar transformer
To train the network in a supervised manner, run below command
```bash
python3 main.py
```

### 5. Evaluate learned correspondences using STEAM
To evaluate the pre-trained network using STEAM, run below command
```bash
python3 evaluate.py
```
Set NUM_FEATURES_PER_POINT = 3 or 4 or 5 depending on the number of features (x, y, z, Doppler, RCS) used for training radar transformer.

Use below commands to get the translation and rotational relative pose errors
```bash
evo_rpe tum steam_gt_pred_trajectory.txt steam_pred_trajectory_xyz.txt --align --correct_scale --delta 1 --delta_unit m -r full --save_results results/rpe_full_xyz.zip --save_plot results/rpe_full_xyz.pdf --plot_mode xy

evo_rpe tum steam_gt_pred_trajectory.txt steam_pred_trajectory_xyz.txt --align --correct_scale --delta 1 --delta_unit m -r trans_part --save_results results/rpe_trans_xyz.zip --save_plot results/rpe_trans_xyz.pdf --plot_mode xy

evo_rpe tum steam_gt_pred_trajectory.txt steam_pred_trajectory_xyz.txt --align --correct_scale --delta 1 --delta_unit m -r rot_part --save_results results/rpe_rot_xyz.zip --save_plot results/rpe_rot_xyz.pdf --plot_mode xy
```

## Base paper citations
```
@INPROCEEDINGS{burnett_rss21,
    title={Radar Odometry Combining Probabilistic Estimation and Unsupervised Feature Learning},
    author={Burnett, Keenan and Yoon, David J and Schoellig, Angela P and Barfoot, Timothy D},
    booktitle={Robotics: Science and Systems},
    year={2021}
}

@ARTICLE{6727494,
  author={Barfoot, Timothy D. and Furgale, Paul T.},
  journal={IEEE Transactions on Robotics}, 
  title={Associating Uncertainty With Three-Dimensional Poses for Use in Estimation Problems}, 
  year={2014},
  volume={30},
  number={3},
  pages={679-693},
  keywords={Uncertainty;Robots;Compounds;Estimation;Covariance matrices;Noise;Probability density function;Exponential maps;homogeneous points;matrix Lie groups;pose uncertainty;transformation matrices},
  doi={10.1109/TRO.2014.2298059}}

@misc{michalczyk2025learningpointcorrespondencesradar,
      title={Learning Point Correspondences In Radar 3D Point Clouds For Radar-Inertial Odometry}, 
      author={Jan Michalczyk and Stephan Weiss and Jan Steinbrener},
      year={2025},
      eprint={2506.18580},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.18580}, 
}

@ARTICLE{10237296,
  author={Lu, Shouyi and Zhuo, Guirong and Xiong, Lu and Zhu, Xichan and Zheng, Lianqing and He, Zihang and Zhou, Mingyu and Lu, Xinfei and Bai, Jie},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={Efficient Deep-Learning 4D Automotive Radar Odometry Method}, 
  year={2024},
  volume={9},
  number={1},
  pages={879-892},
  keywords={Point cloud compression;Radar;Radar cross-sections;Feature extraction;Odometry;Laser radar;Three-dimensional displays;Deep radar odometry;autonomous driving;4D radar},
  doi={10.1109/TIV.2023.3311102}}

@misc{huai2025snailradarlargescalediverse,
      title={SNAIL Radar: A large-scale diverse benchmark for evaluating 4D-radar-based SLAM}, 
      author={Jianzhu Huai and Binliang Wang and Yuan Zhuang and Yiwen Chen and Qipeng Li and Yulong Han},
      year={2025},
      eprint={2407.11705},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2407.11705}, 
}

```
