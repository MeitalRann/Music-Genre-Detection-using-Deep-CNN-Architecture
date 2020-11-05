import numpy as np
import os
import datetime
import random
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import get_images as get_images


# set seed:
seed_value = 1
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
from tensorflow.keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


def create_model(n_classes):
    model = keras.Sequential(
        [keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid', input_shape=(204, 204, 3)),
         keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
         keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid'),
         keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
         keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'),
         keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
         keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'),
         keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
         keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'),
         keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
         keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'),
         keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
         keras.layers.Flatten(),
         keras.layers.Dense(units=512, activation='relu'),
         keras.layers.Dropout(rate=0.3),
         keras.layers.Dense(units=256, activation='relu'),
         keras.layers.Dropout(rate=0.3),
         keras.layers.Dense(units=75, activation='relu'),
         keras.layers.Dropout(rate=0.3),
         keras.layers.Dense(units=n_classes, activation='softmax')
         ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(log_dir, checkpoint_dir, n_classes, train_batches, valid_batches):
        model = create_model(n_classes)
        model.summary()

        my_callbacks = [
            keras.callbacks.EarlyStopping(patience=4),
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir,
                save_weights_only=True),
            keras.callbacks.TensorBoard(log_dir=log_dir)]

        model.fit(x=train_batches, validation_data=valid_batches, epochs=50, verbose=2,  callbacks=my_callbacks)
        return model


def main(mode, n_classes, im_size, prj_dir, feat_dir, classes, batch_size=50):
    print('Getting data...')
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)

    images = get_images.main(feat_dir+r'\\train')
    train_datagen.fit(images)
    valid_datagen.fit(images)
    test_datagen.fit(images)
    del images

    train_batches = train_datagen.flow_from_directory(feat_dir+r'\\train',
                                                      target_size=im_size,
                                                      batch_size=batch_size,
                                                      classes=classes)

    valid_batches = valid_datagen.flow_from_directory(feat_dir + r'\\valid',
                                                      target_size=im_size,
                                                      batch_size=batch_size,
                                                      classes=classes)

    test_batches = test_datagen.flow_from_directory(feat_dir + r'\\test',
                                                      target_size=im_size,
                                                      batch_size=batch_size,
                                                      classes=classes,
                                                      shuffle=False)

    true_test = test_batches.classes
    test_names = test_batches.filenames
    # create log folder if not exists:
    if not os.path.exists(prj_dir + r'\\logs\\'):
        os.makedirs(prj_dir + r'\\logs\\')
    # create model folder if not exists:
    if not os.path.exists(prj_dir + r'\\model\\'):
        os.makedirs(prj_dir + r'\\model\\')

    log_dir = prj_dir + r"\\logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = prj_dir + r".\\model\\model.{epoch:02d}-{val_loss:.2f}.h5"

    if mode is "train":
        print('Start training...')
        model = train(log_dir, checkpoint_dir, n_classes, train_batches, valid_batches)
        print('Done training!')
        print('Start Testing...')
        pred = model.predict(x=test_batches, verbose=0)
        print('Done testing!')
    if mode is 'test':
        # get saved model:
        print('Import model...')
        list_of_files = glob.glob(prj_dir + r'\model\*')  # * means all if need specific format then *.csv
        latest = max(list_of_files, key=os.path.getctime)
        # Create a new model instance
        model = create_model(n_classes)
        # Load the previously saved weights
        model.load_weights(latest)
        print('Start Testing...')
        pred = model.predict(x=test_batches, verbose=0)
        print('Done testing!')
    return pred, true_test, test_names
