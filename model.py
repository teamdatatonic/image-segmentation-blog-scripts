import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img


class Cityscapes(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        # Used to group the labels for objects into broad categories
        # e.g. (car, truck, bus, ...) -> vehicle
        # Mappings taken from: https://github.com/mcordts/cityscapesScripts
        # /blob/master/cityscapesscripts/helpers/labels.py#L52-L99
        self.id_to_cat = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 1,
            8: 1,
            9: 1,
            10: 1,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 2,
            17: 3,
            18: 3,
            19: 3,
            20: 3,
            21: 4,
            22: 4,
            23: 5,
            24: 6,
            25: 6,
            26: 7,
            27: 7,
            28: 7,
            29: 7,
            30: 7,
            31: 7,
            32: 7,
            33: 7,
            -1: 7,
        }

        def __len__(self):
            return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        # Read in a batch of images and convert them to numpy arrays
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        # Read in the segmentation masks for the above images and convert them to numpy arrays
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            # Convert the class IDs of each pixel in each segmentation mask into group IDs
            # i.e. group the labels for objects into broad categories
            # e.g. (car, truck, bus, ...) -> vehicle
            arr = np.vectorize(self.id_to_cat.get)(img)
            y[j] = np.expand_dims(arr, 2)
        return x, y


def get_model(img_size, num_classes):
    """Define a Keras model using a U-Net-like architecture"""
    inputs = tf.keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket_name",
        dest="bucket_name",
        default="my_bucket",
        type=str,
        help="GCS bucket to use",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        default=15,
        type=int,
        help="Number of epochs to train for",
    )
    args = parser.parse_args()

    # Use a GPU if it's available
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    input_dir = f"/gcs/{args.bucket_name}/leftImg8bit_trainvaltest/leftImg8bit"
    target_dir = f"/gcs/{args.bucket_name}/gtFine_trainvaltest/gtFine"
    # All images and segmentation masks are resized to this size to simplify training
    img_size = (160, 160)
    batch_size = 16

    # Get the file paths for all of the images in our training data
    input_img_paths = sorted(
        [
            os.path.join(root, file)
            for root, _, files in os.walk(f"{input_dir}/train/")
            for file in files
            if file.endswith(".png") and not file.startswith(".")
        ]
    )
    # Get the file paths of the corresponding segmentation masks
    target_img_paths = sorted(
        [
            os.path.join(root, file)
            for root, _, files in os.walk(f"{target_dir}/train/")
            for file in files
            if file.endswith("_labelIds.png") and not file.startswith(".")
        ]
    )

    # Load in the training data and segmentation masks using the helper class
    train_gen = Cityscapes(batch_size, img_size, input_img_paths, target_img_paths)

    # The trained model is saved to this directory
    # This environment variable is set by GCP when using the Vertex AI training service
    MODEL_DIR = os.getenv("AIP_MODEL_DIR")

    with strategy.scope():
        # Build model
        model = get_model(img_size, 8)

        # Configure the model for training.
        # We use the "sparse" version of categorical_crossentropy
        # because our target data is integers.
        model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    # Train a model and save it to GCS
    model.fit(train_gen, epochs=args.epochs)
    model.save(MODEL_DIR)
