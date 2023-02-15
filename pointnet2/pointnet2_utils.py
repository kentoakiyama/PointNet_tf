import tensorflow as tf

from utils.layers import NonLinear

def fathest_point_sampling(xyz, n_points):
    """
    xyz: tensor [batch size, num points, num dimension]
    n_points: number of sampling points
    """
    def _gen_idx(b):
        return [b, added_id[b]]

    xyz_shape = tf.shape(xyz)
    B = xyz_shape[0]
    N = xyz_shape[1]
    D = xyz_shape[2]
    centroid_ids = tf.random.uniform([B, 1], minval=0, maxval=N-1, dtype=tf.int32)
    mask = tf.ones([B, N], dtype=tf.float32)
    for i in range(n_points-1):
        added_id = centroid_ids[:, i]
        added_id_tf = tf.stack([tf.range(B), added_id], axis=1) 
        added_point = tf.reshape(tf.gather_nd(xyz, added_id_tf), [B, 1, D])
        dist = tf.math.reduce_euclidean_norm(xyz - added_point, axis=2)
        dist = dist * mask
        max_d_idx = tf.argmax(dist, axis=1, output_type=tf.int32)[..., tf.newaxis]

        centroid_ids = tf.concat([centroid_ids, max_d_idx], axis=1)
        mask = tf.math.minimum(dist * mask * 10e+10, mask)
    return centroid_ids


def get_point_from_idx(xyz, idx):
    """
    xyz: tensor [B, N, D]
    idx: tensor [B, n_points] or [B, n_points, n_samples]

    return:
        [B, n_points, D] or [B, n_points, n_samples, D]
    """
    idx_shape = tf.shape(idx)
    if idx_shape.shape == 2:
        B, n = idx_shape[0], idx_shape[1]
        b_idx = tf.tile(tf.range(B, dtype=tf.int32)[..., tf.newaxis], [1, n])[..., tf.newaxis] # [B, n, 1]
        idx = idx[..., tf.newaxis]  # [B, n, 1]
        idx = tf.concat([b_idx, idx], axis=2)
    elif idx_shape.shape == 3:
        B, np, ns = idx_shape[0], idx_shape[1], idx_shape[2]
        b_idx = tf.tile(tf.reshape(tf.range(B, dtype=tf.int32), [B, 1, 1]), [1, np, ns])[..., tf.newaxis] # [B, np, ns, 1]
        idx = idx[..., tf.newaxis]  # [B, n, 1]
        idx = tf.concat([b_idx, idx], axis=3)
    return tf.gather_nd(xyz, idx)


def query_ball_point(radius, n_samples, xyz, cent_xyz):
    """
    radius:
    n_samples:
    xyz: [B, N, D]
    cent_xyz: [B, n, D]

    return:
        [B, n_points, n_samples]
    """
    xyz_shape = tf.shape(xyz)
    B = xyz_shape[0]
    N = xyz_shape[1]
    D = xyz_shape[2]

    n = tf.shape(cent_xyz)[1]

    # B, N, D = xyz.shape
    # _, n, _ = cent_xyz.shape
    xyz = tf.reshape(xyz, [B, 1, N, D])
    cent_xyz = tf.reshape(cent_xyz, [B, n, 1, D])
    dist = tf.math.reduce_euclidean_norm(tf.tile(xyz, [1, n, 1, 1]) - cent_xyz, axis=3)
    r_mask = tf.ones([B, n, N]) * radius**2
    dist = tf.math.minimum(dist, r_mask)
    return tf.argsort(dist)[:, :, :n_samples]


def sample_and_group(xyz: tf.Tensor, n_points: int, n_samples: int, radius: float, points: tf.Tensor=None):
    """
    xyz: [B, N, D]
    n_points:
    n_sample:
    radius:

    return:
        [[B, n_samples, 3], [B, n_points, n_samples, D]]
    """
    cent_idx = fathest_point_sampling(xyz, n_points)
    cent_xyz = get_point_from_idx(xyz, cent_idx)

    group_idx = query_ball_point(radius, n_samples, xyz, cent_xyz)
    group_xyz = get_point_from_idx(xyz, group_idx)
    
    if points is not None:
        group_points = get_point_from_idx(points, group_idx)
        new_points = tf.concat([group_xyz, group_points], axis=-1)
    else:
        new_points = group_xyz
    return cent_xyz, new_points

def sample_and_group_all(xyz: tf.Tensor, points: tf.Tensor=None):
    """
    xyz: [B, N, C]
    points: [B, N, D]

    return:
        [[B, n_samples, 3], [B, n_points, n_samples, D]]
    """
    xyz_shape = tf.shape(xyz)
    B = xyz_shape[0]
    N = xyz_shape[1]
    C = xyz_shape[2]

    # B, N, C = xyz.shape

    new_xyz = tf.zeros([B, 1, C])
    grouped_xyz = tf.reshape(xyz, [B, 1, N, C])
    if points is not None:
        points = tf.reshape(points, [B, 1, N, -1])
        new_points = tf.concat([grouped_xyz, points], axis=-1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


class SetAbstraction(tf.keras.Model):
    def __init__(self, n_points: int, n_samples: int, radius: float, mlps, activation: str='relu', batchnormalization: bool=True, group_all: bool=False):
        super(SetAbstraction, self).__init__()
        self.n_points = n_points
        self.n_samples = n_samples
        self.radius = radius
        self.group_all = group_all
        self.mlp_layers = []
        for mlp in mlps:
            self.mlp_layers.append(NonLinear(mlp, activation, batchnormalization=batchnormalization))
    
    def call(self, xyz, points):
        """
        xyz: tensor [B, N, 3]
        points: tensor [B, N, D]
        
        return:
            [B, n_point, 3]
            [B, n_point, D]
        """

        if self.group_all:
            group_xyz, group_points = sample_and_group_all(xyz, points)
        else:
            group_xyz, group_points = sample_and_group(xyz, self.n_points, self.n_samples, self.radius, points)

        for mlp in self.mlp_layers:
            group_points = mlp(group_points)

        group_points = tf.reduce_max(group_points, axis=2)
        return group_xyz, group_points