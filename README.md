# PointNet

The network can handle the point cloud data such as mesh data. Prior to the publication of this paper (2017), many researchers transform such data to 3D voxel grid or collections of images.

## Features

To handle the point cloud data, there are three main features,
- Unordered: Unlike the images, the order of data is not meaningful in point cloud. So the output of 3D N point sets needs to be invariant to N! permutation of data.
- Interaction among points: This means the points are not isolated, and the neiboring points form the meaningful relations.
- Invariance  under  transformations: The output of the point cloud shoud be invariant to certain transformation. For example, the output must be invariant to data where all points have been rotated or moved.

## Usage
The example for usage is shown below.

There are two models, one is for classification and the other is for segmentation task. 

```python
# classification task
model = PointNet(num_points, num_out, activation, out_activation, batchnormalization=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=train_gen,
                    epochs=100,
                    steps_per_epoch=len(train_gen),
                    validation_data=val_gen,
                    validation_steps=len(val_gen))

# segmentation task
model = PointNetSeg(num_points, num_out, activation, out_activation, batchnormalization=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=train_gen,
                    epochs=100,
                    steps_per_epoch=len(train_gen),
                    validation_data=val_gen,
                    validation_steps=len(val_gen))
```

You have five options for training.

- num_points: number of points input to network (1024 in original paper)
- num_out: number of output (if ModelNet10, num_out is 10)
- activation: activation function
- out_activation: activation function for output layer (You can choose in 'softmax', 'sigmoid', and 'linear'.)
- batchnormalization: True or False (default: True)


## Environment

- TensorFlow (ver. >= 2.0)

## Reference
- [R. Q. Charles, et al., PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation, 2017](https://arxiv.org/pdf/1612.00593.pdf)