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
    validation_size = 0.2
    batch_size = 32
    num_points = 1024
    activation = 'relu'
    epochs = 200
    init_lr = 1e-3
    min_lr = 1e-6
    drop_rate = 0.5
    lr_patience = 10


    logger.info('Start training!!')
    logger.info('Setting')
    logger.info(f'seed: {seed}')
    logger.info(f'validation_size: {validation_size}')
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'num_points: {num_points}')
    logger.info(f'activation: {activation}')
    logger.info(f'epochs: {epochs}')
    logger.info(f'initial learning rate: {init_lr}')
    logger.info(f'minimum learning rate: {min_lr}')
    logger.info(f'drop rate: {drop_rate}')
    logger.info(f'patience for learning rate schedule: {lr_patience}')

    # -------------------------------------------------------------------------------------
    # generating the dataset
    file_pattern = list(Path(root_dir).glob('**/train/*.*off'))
    label_pattern = [path.name.split('_')[0] for path in file_pattern]
    labels = list(set(label_pattern))

    train_pattern, val_pattern, _, _ = \
        train_test_split(file_pattern, label_pattern, test_size=validation_size, stratify=label_pattern, random_state=seed)
    train_pattern, val_pattern = list(train_pattern), list(val_pattern)

    test_pattern = list(Path(root_dir).glob('**/test/*.*off'))

    train_gen = ModelNetDataLoader(train_pattern, batch_size=batch_size, labels=labels, num_points=num_points, augment=True)
    val_gen = ModelNetDataLoader(val_pattern, batch_size=batch_size, labels=labels, num_points=num_points, augment=False)
    test_gen = ModelNetDataLoader(test_pattern, batch_size=batch_size, labels=labels, num_points=num_points, augment=False)
    
    logger.info(f'Number of labels: {len(labels)}')
    logger.info(f'Training size: {len(train_pattern)}')
    logger.info(f'Validation size: {len(val_pattern)}')
    logger.info(f'Test size: {len(test_pattern)}')
    
    # -------------------------------------------------------------------------------------
    # training
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=drop_rate,
                                                     patience=lr_patience, min_lr=min_lr)
    model = PointNet(num_points, len(train_gen.labels), activation)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_gen,
              epochs=epochs,
              steps_per_epoch=len(train_gen),
              validation_data=val_gen,
              validation_steps=len(val_gen),
              callbacks=[reduce_lr],
              workers=6,
              max_queue_size=40)
    
    training_scores = model.evaluate(train_gen)
    val_scores = model.evaluate(val_gen)
    test_scores = model.evaluate(test_gen)
    
    logger.info(f'Training accuracy  : {training_scores:.4f}')
    logger.info(f'Validation accuracy: {val_scores:.4f}')
    logger.info(f'Test accuracy      : {test_scores:.4f}')


if __name__ == '__main__':
    main()


    