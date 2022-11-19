import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path


class ModelNetDataLoader(tf.keras.utils.Sequence):
    def __init__(self, file_pattern, batch_size: int, labels, num_points: int=1024, augment: bool=False):
        self.file_pattern = file_pattern
        self.type = type
        self.batch_size = batch_size
        self.num_points = num_points
        self.augment = augment
        self.labels = labels

        self.cache_data = {}


    def __len__(self):
        return np.ceil(len(self.file_pattern) / self.batch_size).astype('int')

    def _read_off(self, file):
        first_line = file.readline().strip()
        if 'OFF' == first_line:
            n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
        elif 'OFF' in first_line:
            n_verts, n_faces, __ = tuple([int(s) for s in first_line[3:].split(' ')])
        else:
            raise('Not a valid OFF header')
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        return verts, faces

    def _triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def _sample_point(self, pt1, pt2, pt3):
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s) * pt2[i] + (1-t) * pt3[i]
        return (f(0), f(1), f(2))

    def _jittering(self, pointcloud, sigma: float=0.02):
        return pointcloud + np.random.normal(0, sigma, pointcloud.shape)

    def _rotation(self, pointcloud):
        theta = 2 * np.pi * np.random.rand()

        cos = np.cos(theta)
        sin = np.sin(theta)

        rot_mtx = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        return pointcloud @ rot_mtx

    def _data_parser(self, file_path):
        pointcloud = self._get_pointcloud(file_path)
        
        if self.augment:
            # rotation
            pointcloud = self._rotation(pointcloud)

            # jittering
            pointcloud = self._jittering(pointcloud, sigma=0.02)
        return pointcloud

    def _get_pointcloud(self, file_path):
        if file_path not in self.cache_data.keys():
            with open(file_path, 'r') as f:
                mesh = self._read_off(f)

            verts, faces = mesh
            areas = np.zeros((len(faces)))
            verts = np.array(verts)
            
            for i in range(len(areas)):
                areas[i] = (self._triangle_area(verts[faces[i][0]],
                                        verts[faces[i][1]],
                                        verts[faces[i][2]]))
            
            sampled_faces = (random.choices(faces, 
                                            weights=areas,
                                            k=self.num_points))

            pointcloud = np.zeros((self.num_points, 3))

            # sample points on chosen faces for the point cloud of size 'k'
            for i in range(len(sampled_faces)):
                pointcloud[i] = (self._sample_point(verts[sampled_faces[i][0]],
                                                    verts[sampled_faces[i][1]],
                                                    verts[sampled_faces[i][2]]))

            # standardization
            pointcloud = (pointcloud - np.mean(pointcloud, axis=0)) / (np.std(pointcloud, axis=0) + 1e-8)
            self.cache_data[file_path] = pointcloud
        
        else:
            pointcloud = self.cache_data[file_path].copy()

        return pointcloud
    
    def _get_label(self, file_path):
        # get label
        label = Path(file_path).name.split('_')[0]
        onehot_label = [1 if c == label else 0 for i, c in enumerate(self.labels)]
        return onehot_label

    def __getitem__(self, idx):
        batch_file_pattern = self.file_pattern[idx*self.batch_size:(idx+1)*self.batch_size]
        
        batch_x = np.array([self._data_parser(path) for path in batch_file_pattern])
        batch_y = np.array([self._get_label(path) for path in batch_file_pattern])
        return batch_x, batch_y


class ModelNetDataLoaderProccessed(tf.keras.utils.Sequence):
    def __init__(self, file_pattern, batch_size: int, labels, num_points: int=1024, augment: bool=False):
        self.file_pattern = file_pattern
        self.type = type
        self.batch_size = batch_size
        self.num_points = num_points
        self.augment = augment
        self.labels = labels


    def __len__(self):
        return np.ceil(len(self.file_pattern) / self.batch_size).astype('int')

    def _jittering(self, pointcloud, sigma: float=0.02):
        return pointcloud + np.random.normal(0, sigma, pointcloud.shape)

    def _rotation(self, pointcloud):
        theta = 2 * np.pi * np.random.rand()

        cos = np.cos(theta)
        sin = np.sin(theta)

        rot_mtx = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        return pointcloud @ rot_mtx

    def _data_parser(self, file_path):
        pointcloud = np.loadtxt(file_path).astype('float32')
        pointcloud = pointcloud[:, :3]

        # standardization
        pointcloud = (pointcloud - np.mean(pointcloud, axis=0)) / np.std(pointcloud, axis=0)
        
        if self.augment:
            # rotation
            pointcloud = self._rotation(pointcloud)

            # jittering
            pointcloud = self._jittering(pointcloud, sigma=0.02)
        return pointcloud

    def _get_label(self, file_path):
        # get label
        label = Path(file_path).name.split('_')[0]
        onehot_label = [1 if c == label else 0 for i, c in enumerate(self.labels)]
        return onehot_label

    def __getitem__(self, idx):
        batch_file_pattern = self.file_pattern[idx*self.batch_size:(idx+1)*self.batch_size]
        
        batch_x = np.array([self._data_parser(path) for path in batch_file_pattern])
        batch_y = np.array([self._get_label(path) for path in batch_file_pattern])
        return batch_x, batch_y