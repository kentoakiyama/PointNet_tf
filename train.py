import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from logging import getLogger, Formatter, DEBUG, StreamHandler

from pointnet.pointnet import PointNet
from examples.dataloader import ModelNetDataLoader

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


def main():
    logger = custom_logger(__name__)
    root_dir = r"D:\Projects\deeplearning\PointNet_tf\data\ModelNet40\ModelNet40"
    
    seed = 0
    batch_size = 32
    num_points = 1024
    activation = 'relu'
    epochs = 300
    init_lr = 1e-3
    min_lr = 1e-6
    decay_rate = 0.7
    decay_steps = 10000


    logger.info('Start training!!')
    logger.info('Setting')
    logger.info(f'seed: {seed}')
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'num_points: {num_points}')
    logger.info(f'activation: {activation}')
    logger.info(f'epochs: {epochs}')
    logger.info(f'initial learning rate: {init_lr}')
    logger.info(f'minimum learning rate: {min_lr}')
    logger.info(f'decay rate: {decay_rate}')
    logger.info(f'decay steps: {decay_steps}')

    # -------------------------------------------------------------------------------------
    # generating the dataset
    train_pattern = list(Path(root_dir).glob('**/train/*.*off'))
    label_pattern = [path.name.split('_')[0] for path in train_pattern]
    labels = list(set(label_pattern))


    test_pattern = list(Path(root_dir).glob('**/test/*.*off'))

    train_gen = ModelNetDataLoader(train_pattern, batch_size=batch_size, labels=labels, num_points=num_points, augment=True)
    test_gen = ModelNetDataLoader(test_pattern, batch_size=batch_size, labels=labels, num_points=num_points, augment=False)
    
    logger.info(f'Number of labels: {len(labels)}')
    logger.info(f'Training size: {len(train_pattern)}')
    logger.info(f'Test size: {len(test_pattern)}')
    
    # -------------------------------------------------------------------------------------
    # training
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        init_lr, decay_steps, decay_rate, staircase=True, name=None
    )
    model = PointNet(num_points, len(train_gen.labels), activation)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_gen,
              epochs=epochs,
              steps_per_epoch=len(train_gen),
              validation_data=test_gen,
              validation_steps=len(test_gen),
              workers=6,
              max_queue_size=40)
    
    training_scores = model.evaluate(train_gen)
    test_scores = model.evaluate(test_gen)
    
    logger.info(f'Training accuracy  : {training_scores[1]:.4f}')
    logger.info(f'Test accuracy      : {test_scores[1]:.4f}')


if __name__ == '__main__':
    main()


    