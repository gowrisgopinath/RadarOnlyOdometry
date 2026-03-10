import torch
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def wrap_angle(theta):
    return torch.atan2(torch.sin(theta), torch.cos(theta))

def se2_apply_transform(points_xyz, yaw, t_xy):
    # points_xyz: (B,N,3), yaw: (B,1), t_xy: (B,2)
    ca = torch.cos(yaw).view(-1, 1, 1)   # (B,1,1)
    sa = torch.sin(yaw).view(-1, 1, 1)   # (B,1,1)
    t_xy = t_xy.view(-1, 1, 2)           # (B,1,2)

    x = points_xyz[..., 0:1]             # (B,N,1)
    y = points_xyz[..., 1:2]             # (B,N,1)
    z = points_xyz[..., 2:3]             # (B,N,1)

    xr = ca * x - sa * y                 # (B,N,1)
    yr = sa * x + ca * y                 # (B,N,1)
    xy_rt = torch.cat([xr, yr], dim=-1) + t_xy  # (B,N,2)

    return torch.cat([xy_rt, z], dim=-1) 

def apply_se2_inverse(points_xyz, yaw, t_xy):
    # T^(-1): R(-yaw) * (x - t)
    ca = torch.cos(yaw).view(-1, 1, 1)  # (B,1,1)
    sa = torch.sin(yaw).view(-1, 1, 1)  # (B,1,1)

    xy = points_xyz[..., :2] - t_xy.view(-1, 1, 2)  # (B,N,2)

    x = xy[..., 0:1]  # (B,N,1)
    y = xy[..., 1:2]  # (B,N,1)

    xr =  ca * x + sa * y
    yr = -sa * x + ca * y

    z  = points_xyz[..., 2:3]
    return torch.cat([xr, yr, z], dim=-1)  # (B,N,3)

def se2_compose(yaw_a, t_a, yaw_b, t_b):
    ca, sa = torch.cos(yaw_a), torch.sin(yaw_a)     # (B,1)
    tx = ca * t_b[..., 0:1] - sa * t_b[..., 1:2]    # (B,1)
    ty = sa * t_b[..., 0:1] + ca * t_b[..., 1:2]
    t = torch.cat([tx, ty], dim=-1) + t_a           # (B,2)
    yaw = wrap_angle(yaw_a + yaw_b)                 # (B,1)
    return yaw, t

def slerp_se3(ts, key_ts, T_list):
    # SE(3) interpolation: linear xyz + SLERP quat interpolation
    if ts <= key_ts[0]:  return T_list[0]
    if ts >= key_ts[-1]: return T_list[-1]
    i = np.searchsorted(key_ts, ts) - 1
    t0, t1 = key_ts[i], key_ts[i+1]
    a = (ts - t0) / (t1 - t0)
    T0, T1 = T_list[i], T_list[i+1]
    p0, p1 = T0[:3,3], T1[:3,3]
    R0, R1 = T0[:3,:3], T1[:3,:3]

    slerp = Slerp([0.0, 1.0], R.from_matrix([R0, R1]))
    Rot_a = slerp([a])[0].as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = Rot_a
    T[:3, 3] = (1-a)*p0 + a*p1
    return T

def se2_from_T(T):
    dx, dy = T[0, 3], T[1, 3]
    yaw = np.arctan2(T[1, 0], T[0, 0])
    return np.array([dx, dy, yaw], dtype=np.float32)

def se3_vec6_to_T(vec6):
    tx, ty, tz, r, p, y = [float(v) for v in vec6]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler('xyz', [r, p, y], degrees=False).as_matrix()
    T[:3, 3]  = [tx, ty, tz]
    return T

def warp_pc_se3(pc_xyz, vec6):
    """
    pc_xyz  : (N,3) numpy
    vec6 : [tx,ty,tz, roll,pitch,yaw]
    """
    t = vec6[:3]
    rpy = vec6[3:6]
    Rm = R.from_euler('xyz', rpy, degrees=False).as_matrix()
    return (pc_xyz @ Rm.T) + t

#auto grad safe SE(3) helpers
'''
def euler_to_R(e):  # e: [roll, pitch, yaw]
    cx, cy, cz = torch.cos(e[:,0]), torch.cos(e[:,1]), torch.cos(e[:,2])
    sx, sy, sz = torch.sin(e[:,0]), torch.sin(e[:,1]), torch.sin(e[:,2])
    R = e.new_zeros(e.shape[0], 3, 3)
    R[:,0,0] =  cy*cz;            R[:,0,1] = -cy*sz;            R[:,0,2] =  sy
    R[:,1,0] =  cx*sz + cz*sx*sy; R[:,1,1] =  cx*cz - sx*sy*sz; R[:,1,2] = -cy*sx
    R[:,2,0] =  sx*sz - cx*cz*sy; R[:,2,1] =  cz*sx + cx*sy*sz; R[:,2,2] =  cx*cy
    return R

def R_to_euler(R):
    sy = R[:, 0, 2].clamp(-1.0, 1.0)
    pitch = torch.asin(sy)
    cy = torch.cos(pitch)
    near = (cy.abs() < 1e-6)

    yaw_n  = torch.atan2(-R[:, 0, 1], R[:, 0, 0])
    roll_n = torch.atan2(-R[:, 1, 2], R[:, 2, 2])

    yaw_g  = torch.atan2( R[:, 1, 0], R[:, 1, 1])
    roll_g = torch.zeros_like(pitch)

    yaw  = torch.where(near, yaw_g,  yaw_n)
    roll = torch.where(near, roll_g, roll_n)

    return torch.stack([wrap_angle(roll), wrap_angle(pitch), wrap_angle(yaw)], dim=1)

def apply_corr_on_init_se3(init6, corr6):
    t_i, e_i = init6[:, :3], init6[:, 3:]
    t_c, e_c = corr6[:, :3], corr6[:, 3:]
    R_i, R_c = euler_to_R(e_i), euler_to_R(e_c)
    R_f = torch.bmm(R_c, R_i)
    t_f = torch.bmm(R_c, t_i.unsqueeze(-1)).squeeze(-1) + t_c
    e_f = R_to_euler(R_f)
    return torch.cat([t_f, e_f], dim=1)

def warp_pc_se3_torch(pc_xyz, vec6):
    t = vec6[:, :3].unsqueeze(1) # (B,1,3)
    e = vec6[:, 3:]
    R = euler_to_R(e)         # (B,3,3)
    return torch.bmm(pc_xyz, R.transpose(1,2)) + t'''


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.pi) % (2 * torch.pi) - torch.pi

def euler_to_R_XYZ(angles: torch.Tensor) -> torch.Tensor:
    rx, ry, rz = angles.unbind(-1)

    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    # Rx
    r00 = torch.ones_like(cx); r01 = torch.zeros_like(cx); r02 = torch.zeros_like(cx)
    r10 = torch.zeros_like(cx); r11 =  cx;                  r12 = -sx
    r20 = torch.zeros_like(cx); r21 =  sx;                  r22 =  cx

    # Rx @ Ry
    # [[cy, -sy*cx,  sy*sx],
    #  [sy,  cy*cx, -cy*sx],
    #  [ 0,      sx,     cx]]
    m00 =  cy
    m01 = -sy * cx
    m02 =  sy * sx

    m10 =  sy
    m11 =  cy * cx
    m12 = -cy * sx

    m20 =  rx.new_zeros(rx.shape)
    m21 =  sx
    m22 =  cx

    # (Rx @ Ry) @ Rz
    # [[m00*cz + m01*sz,  -m00*sz + m01*cz,  m02],
    #  [m10*cz + m11*sz,  -m10*sz + m11*cz,  m12],
    #  [m20*cz + m21*sz,  -m20*sz + m21*cz,  m22]]
    R = angles.new_zeros(angles.shape[0], 3, 3)
    R[:, 0, 0] =  m00 * cz + m01 * sz
    R[:, 0, 1] = -m00 * sz + m01 * cz
    R[:, 0, 2] =  m02

    R[:, 1, 0] =  m10 * cz + m11 * sz
    R[:, 1, 1] = -m10 * sz + m11 * cz
    R[:, 1, 2] =  m12

    R[:, 2, 0] =  m20 * cz + m21 * sz
    R[:, 2, 1] = -m20 * sz + m21 * cz
    R[:, 2, 2] =  m22
    return R


def R_to_euler_XYZ(R: torch.Tensor) -> torch.Tensor:
    sy = R[:, 0, 2].clamp(-1.0, 1.0)
    pitch = torch.asin(sy)
    cy = torch.cos(pitch)
    near = (cy.abs() < 1e-6)

    yaw_n  = torch.atan2(-R[:, 0, 1], R[:, 0, 0])
    roll_n = torch.atan2(-R[:, 1, 2], R[:, 2, 2])

    yaw_g  = torch.atan2(R[:, 1, 0], R[:, 1, 1])
    roll_g = torch.zeros_like(pitch)

    yaw  = torch.where(near, yaw_g,  yaw_n)
    roll = torch.where(near, roll_g, roll_n)

    e = torch.stack([wrap_angle(roll), wrap_angle(pitch), wrap_angle(yaw)], dim=1)
    return e

def apply_corr_on_init_se3_XYZ(init6: torch.Tensor, corr6: torch.Tensor) -> torch.Tensor:
    t_i, e_i = init6[:, :3], init6[:, 3:]
    t_c, e_c = corr6[:, :3], corr6[:, 3:]

    R_i = euler_to_R_XYZ(e_i)
    R_c = euler_to_R_XYZ(e_c)

    R_f = R_c @ R_i
    t_f = (R_c @ t_i.unsqueeze(-1)).squeeze(-1) + t_c

    e_f = R_to_euler_XYZ(R_f)
    e_f = wrap_angle(e_f)
    return torch.cat([t_f, e_f], dim=1)


def warp_pc_se3_XYZ(pc_xyz: torch.Tensor, vec6: torch.Tensor) -> torch.Tensor:
    t = vec6[:, :3].unsqueeze(1)        # (B,1,3)
    Rm = euler_to_R_XYZ(vec6[:, 3:])    # (B,3,3)
    return pc_xyz @ Rm.transpose(1, 2) + t
