import mindspore
from mindspore import nn, ops
import mindspore.ops.functional as F
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * ops.matmul(src, ops.transpose(dst, (0, 2, 1)))
    dist += ops.expand_dims(ops.reduce_sum(src ** 2, -1), -1)
    dist += ops.expand_dims(ops.reduce_sum(dst ** 2, -1), 1)
    return dist

def index_points(points, idx):
    """
    输入：
        points: 输入点数据，形状为 [B, N, C]
        idx: 采样的索引，形状为 [B, S] 或 [B, S, nsample]
    返回：
        new_points: 索引后的点数据，形状为 [B, S, C] 或 [B, S, nsample, C]
    """
    B = points.shape[0]
    idx = idx.astype(mindspore.int32)

    if len(idx.shape) == 2:
        S = idx.shape[1]
        batch_indices = ops.tile(ops.arange(B, dtype=mindspore.int32).view(B, 1), (1, S))
        new_points = points[batch_indices, idx, :]  # [B, S, C]
    elif len(idx.shape) == 3:
        S, nsample = idx.shape[1], idx.shape[2]
        batch_indices = ops.tile(ops.arange(B, dtype=mindspore.int32).view(B, 1, 1), (1, S, nsample))
        new_points = points[batch_indices, idx, :]  # [B, S, nsample, C]
    else:
        raise ValueError("idx 的维度不正确，应为 2 或 3")

    return new_points


def farthest_point_sample(xyz, npoint):
    B, N, C = xyz.shape
    centroids = ops.zeros((B, npoint), mindspore.int32)
    distance = ops.fill(mindspore.float32, (B, N), 1e10)
    farthest = ops.randint(0, N, (B,), dtype=mindspore.int32)
    batch_indices = ops.arange(B, dtype=mindspore.int32)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = ops.reduce_sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance = ops.select(mask, dist, distance)
        farthest = ops.ArgMaxWithValue(-1)(distance)[0]

    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = ops.arange(N, dtype=mindspore.int32).view(1, 1, N).repeat(B, 0).repeat(S, 1)
    sqrdists = square_distance(new_xyz, xyz)

    group_idx = ops.select(sqrdists > radius ** 2, ops.fill(mindspore.int32, group_idx.shape, N), group_idx)

    # Convert group_idx to float32 for sorting
    group_idx = group_idx.astype(mindspore.float32)

    # Perform sorting
    sorted_idx = ops.Sort(axis=-1)(group_idx)[0][:, :, :nsample]

    # Convert back to int32
    sorted_idx = sorted_idx.astype(mindspore.int32)

    group_first = sorted_idx[:, :, 0].view(B, S, 1).repeat(nsample,2)
    mask = sorted_idx == N
    sorted_idx = ops.select(mask, group_first, sorted_idx)

    return sorted_idx



def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - ops.expand_dims(new_xyz, 2)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = ops.concat((grouped_xyz_norm, grouped_points), -1)
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    B, N, C = xyz.shape
    new_xyz = ops.zeros((B, 1, C), xyz.dtype)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = ops.concat((grouped_xyz, points.view(B, 1, N, -1)), -1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Cell):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, remove_last=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.CellList()
        self.mlp_bns = nn.CellList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1,has_bias=True))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.remove_last = remove_last

    def construct(self, xyz, points):
        xyz = ops.transpose(xyz, (0, 2, 1))
        if points is not None:
            points = ops.transpose(points, (0, 2, 1))

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = ops.transpose(new_points, (0, 3, 2, 1))
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = ops.reduce_max(new_points, 2)
        new_xyz = ops.transpose(new_xyz, (0, 2, 1))
        if self.remove_last:
            return new_points
        else:
            return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Cell):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.CellList()
        self.bn_blocks = nn.CellList()
        for i in range(len(mlp_list)):
            convs = nn.CellList()
            bns = nn.CellList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def construct(self, xyz, points):
        xyz = ops.transpose(xyz, (0, 2, 1))
        if points is not None:
            points = ops.transpose(points, (0, 2, 1))

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= ops.expand_dims(new_xyz, 2)

            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = ops.concat((grouped_points, grouped_xyz), -1)
            else:
                grouped_points = grouped_xyz

            grouped_points = ops.transpose(grouped_points, (0, 3, 2, 1))
            for j, conv in enumerate(self.conv_blocks[i]):
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = ops.reduce_max(grouped_points, 2)
            new_points_list.append(new_points)

        new_xyz = ops.transpose(new_xyz, (0, 2, 1))
        new_points_concat = ops.concat(new_points_list, 1)
        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Cell):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.CellList()
        self.mlp_bns = nn.CellList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def construct(self, xyz1, xyz2, points1, points2):
        xyz1 = ops.transpose(xyz1, (0, 2, 1))
        xyz2 = ops.transpose(xyz2, (0, 2, 1))
        points2 = ops.transpose(points2, (0, 2, 1))
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = ops.tile(points2, (1, N, 1))
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = ops.Sort(axis=-1)(dists)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = ops.reduce_sum(dist_recip, axis=2, keepdims=True)
            weight = dist_recip / norm
            interpolated_points = ops.reduce_sum(index_points(points2, idx) * ops.expand_dims(weight, -1), axis=2)

        if points1 is not None:
            points1 = ops.transpose(points1, (0, 2, 1))
            new_points = ops.concat((points1, interpolated_points), -1)
        else:
            new_points = interpolated_points

        new_points = ops.transpose(new_points, (0, 2, 1))
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points