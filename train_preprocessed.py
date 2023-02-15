import os
os.environ['PYTHONHAHSSEED'] = '0'
import random
import argparse
import numpy as np
import tensorflow as tf
from logging import getLogger, Formatter, DEBUG, StreamHandler

from pointnet.pointnet import PointNet
from pointnet2.pointnet2_cls import PointNetClsSSG
from examples.dataloader import ModelNetDataLoaderProccessed

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def custom_logger(name):
    logger = getLogger(name)
    logger.setLevel(DEBUG)

    handler = StreamHandler()
    formatter = Formatter('%(asctime)s  [%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

def reset_random_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
  
def main(model_name: str, data_dir: str, classes: int, batch_size: int, num_points: int, activation: str, epochs: int, init_lr: float, decay_steps: float, decay_rate: float, seed: int):
    logger = custom_logger(__name__)
    
    logger.info('Start training!!')
    logger.info('Setting')
    logger.info(f'Model: {model_name}')
    logger.info(f'seed: {seed}')
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'num_points: {num_points}')
    logger.info(f'activation: {activation}')
    logger.info(f'epochs: {epochs}')
    logger.info(f'initial learning rate: {init_lr}')
    logger.info(f'decay steps: {decay_steps}')
    logger.info(f'decay rate: {decay_rate}')

    # set seed
    reset_random_seeds(seed)

    # -------------------------------------------------------------------------------------
    # generating the dataset
    train_gen = ModelNetDataLoaderProccessed(data_dir, split='train', classes=classes, batch_size=batch_size, num_points=num_points, augment=True)
    test_gen = ModelNetDataLoaderProccessed(data_dir, split='test', classes=classes, batch_size=batch_size, num_points=num_points, augment=False)
    
    logger.info(f'Number of labels: {len(train_gen.labels)}')
    logger.info(f'Training size: {len(train_gen.file_pattern)}')
    logger.info(f'Test size: {len(test_gen.file_pattern)}')
    
    # -------------------------------------------------------------------------------------
    # training
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(init_lr),
        decay_steps=int(decay_steps),
        decay_rate=float(decay_rate),
        staircase=True,
        name=None
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model',
                                                    monitor='val_accuracy',
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode='max',
                                                    verbose=1)

    if model_name == 'pointnet':
        model = PointNet(len(train_gen.labels), activation)
    elif model_name == 'pointnet2ssg':
        model = PointNetClsSSG(len(train_gen.labels), activation=activation)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x=train_gen,
              epochs=epochs,
              steps_per_epoch=len(train_gen),
              validation_data=test_gen,
              validation_steps=len(test_gen),
              callbacks=[checkpoint],
              workers=6,
              max_queue_size=40)
    
    del model

    model = tf.keras.models.load_model('model')
    test_scores = model.evaluate(test_gen)
    
    logger.info(f'Test accuracy: {test_scores[1]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--model', type=str, default='pointnet')
    parser.add_argument('--classes', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--init_lr', type=str, default=1e-3)
    parser.add_argument('--decay_steps', type=str, default=10000)
    parser.add_argument('--decay_rate', type=str, default=0.7)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
  
    main(args.model,
         args.data_dir,
         args.classes,
         args.batch_size,
         args.num_points,
         args.activation,
         args.epochs,
         args.init_lr,
         args.decay_steps,
         args.decay_rate,
         args.seed)


    