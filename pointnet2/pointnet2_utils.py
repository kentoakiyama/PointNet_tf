import tensorflow as tf

from utils.layers import NonLinear

def fathest_point_sampling(xyz, n_points):
    """
    xyz: tensor [batch size, num points, num dimension]
    n_points: number of sampling points
    """
    B, N, D = xyz.shape
    centroid_ids = tf.random.uniform([B, 1], minval=0, maxval=N-1, dtype=tf.int64)
    mask = tf.ones([B, N], dtype=tf.float32)
    for i in range(n_points-1):
        added_id = centroid_ids[:, i]
        added_id_tf = [[b, added_id[b]] for b in range(B)]
        added_point = tf.reshape(tf.gather_nd(xyz, added_id_tf), [B, 1, D])
        dist = tf.math.reduce_euclidean_norm(xyz - added_point, axis=2)
        dist = dist * mask
        max_d_idx = tf.argmax(dist, axis=1)[..., tf.newaxis]

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


    if len(idx.shape) == 2:
        B, n = idx.shape
        b_idx = tf.tile(tf.range(B, dtype=tf.int64)[..., tf.newaxis], [1, n])[..., tf.newaxis] # [B, n, 1]
        idx = idx[..., tf.newaxis]  # [B, n, 1]
        idx = tf.concat([b_idx, idx], axis=2)
    elif len(idx.shape) == 3:
        B, np, ns = idx.shape
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
    B, N, D = xyz.shape
    _, n, _ = cent_xyz.shape
    xyz = tf.reshape(xyz, [B, 1, N, D])
    cent_xyz = tf.reshape(cent_xyz, [B, n, 1, D])
    dist = tf.math.reduce_euclidean_norm(tf.tile(xyz, [1, n, 1, 1]) - cent_xyz, axis=3)
    r_mask = tf.ones([B, D, N]) * radius**2
    dist = tf.math.minimum(dist, r_mask)
    return tf.argsort(dist)[:, :, :n_samples]


def sample_and_group(xyz, n_points, n_samples, radius):
    """
    xyz: [B, N, D]
    n_points:
    n_sample:
    radius:

    return:
        [B, n_points, n_samples, D]
    """
    cent_idx = fathest_point_sampling(xyz, n_points)
    cent_xyz = get_point_from_idx(xyz, cent_idx)

    group_idx = query_ball_point(radius, n_samples, xyz, cent_xyz)
    group_xyz = get_point_from_idx(xyz, group_idx)
    return group_xyz


class SetAbstraction(tf.keras.Model):
    def __init__(self, n_points: int, n_samples: int, radius: float, mlps, activation: str='relu', batchnormalization: bool=True):
        super(SetAbstraction, self).__init__()
        self.n_points = n_points
        self.n_samples = n_samples
        self.radius = radius
        self.mlp_layers = []
        for mlp in mlps:
            self.mlp_layers.append(NonLinear(mlp, activation, batchnormalization=batchnormalization))
    
    def call(self, inputs):
        """
        inputs: tensor [B, N, D]
        
        return:
            [B, n_point, D]
        """
        group_xyz = sample_and_group(inputs, self.n_points, self.n_samples, self.radius)

        for mlp in self.mlp_layers:
            group_xyz = mlp(group_xyz)

        group_xyz = tf.reduce_max(group_xyz, axis=2)
        return group_xyz