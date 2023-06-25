import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label 

def create_mlp_model(cwd = os.path.dirname(__file__)):
    """ 
    Load the MNIST dataset

    shuffle_files=True: The MNIST data is only stored in a single file, 
    but for larger datasets with multiple files on disk, it's good practice to shuffle them when training.

    as_supervised=True: Returns a tuple (img, label) instead of a dictionary {'image': img, 'label': label}

    ds_info: contains information about the MNIST dataset
    """
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    print(ds_info)

    # Training pipeline 
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Evalutation pipeline
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # Create and train model
    # Add Layers
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    # Compile model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    # Train the model (using ds_train)
    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )
    model.summary()
    model.save(cwd + '/mlp_number_guesser.h5')