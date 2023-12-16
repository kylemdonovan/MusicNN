# Names: Kyle Donovan, Philip Hopkins, Marco Scandroglio
# Course: CS 467 Fall 2023
# Project: Top-n Music Genre Classification Neural Network
# GitHub Repo: https://github.com/pdhopkins/CS467_music_NN
# Description: Model definition and training program for the Genre NN

# Model layers were inspired by the Keras documentation,
# https://keras.io/getting_started/intro_to_keras_for_engineers

# imports
import tensorflow as tf
import keras
from keras import layers
import json
import os


def build_model():
    """
    Build a Convolutional Neural Network (CNN) model for genre classification.

    Returns:
        A Keras Model object representing the genre classification model.

    Model Architecture:
        - Input shape: (128, 1292, 1)
        - Rescales input to the range of [0, 1].
        - Convolutional layers with ReLU activation.
        - Max pooling layers.
        - Dropout layers for regularization.
        - Dense layer with softmax activation for genre classification.

    Note:
        The model is designed for a specific input shape (128, 1292, 1).

    Reference:
        - The model architecture is inspired by common practices in image classification
        since it utilizes 2-dimensional matrices as training data.

    Returns:
        A Keras Model object representing the genre classification model.
    """

    number_of_genres = 10
    input = keras.Input(shape=(128, 1292, 1))
    # Rescaling puts everything in the range of [0, 1]
    output = layers.Rescaling(scale=1.0/80,
                              offset=1.0
                              )(input)
    # Convolutional layer makes numerous levels of the tensors
    # Relu activation is a rectified logic unit - makes negatives go to zero
    output = layers.Conv2D(filters=32, kernel_size=(2, 2), activation="relu")(output)
    # Average pooling layer pools every 2x2 to its average
    output = layers.AveragePooling2D(pool_size=(2, 2))(output)
    output = layers.Dropout(0.25)(output)
    output = layers.Conv2D(filters=64, kernel_size=(2, 2), activation="relu")(output)
    output = layers.AveragePooling2D(pool_size=(2, 2))(output)
    output = layers.Dropout(0.25)(output)
    output = layers.Conv2D(filters=128, kernel_size=(2, 2), activation="relu")(output)
    output = layers.AveragePooling2D(pool_size=(2, 2))(output)
    output = layers.Dropout(0.25)(output)
    output = layers.Conv2D(filters=256, kernel_size=(2, 2), activation="relu")(output)
    output = layers.AveragePooling2D(pool_size=(2, 2))(output)
    output = layers.Dropout(0.25)(output)
    output = layers.Conv2D(filters=512, kernel_size=(2, 2), activation="relu")(output)
    output = layers.AveragePooling2D(pool_size=(2, 2))(output)
    # Flatten takes the entire system down to a 1D tensor
    output = layers.Flatten()(output)
    # Dropout randomly sets values to zero, to help with over fitting
    output = layers.Dropout(0.5)(output)
    # The last dense layer provides the "output" of 10 nodes
    output = layers.Dense(number_of_genres, activation="softmax")(output)
    return keras.Model(input, output)


def train_model(path_to_dataset: str, model_name: str) -> None:
    """
    Train a genre classification model using the provided dataset and save the trained model.

    Args:
        path_to_dataset (str): The file path to the dataset for training.
        model_name (str): The name to be used when saving the trained model.

    Returns:
        None

    Notes:
        - The function uses the build_model function to create the neural network architecture.
        - The model is compiled using RMSprop optimizer and sparse categorical crossentropy loss.
        - The training is performed for 10 epochs.

    Example:
        train_model("path/to/dataset.tfrecord", "my_trained_model")
    """

    path_to_val_set = "../dataset_file"
    # generate the genre and validation datasets
    genre_dataset = tf.data.Dataset.load(path_to_dataset)
    val_dataset = tf.data.Dataset.load(path_to_val_set)
    model = build_model()
    model.compile(optimizer="RMSprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="Accuracy")])
    model.fit(genre_dataset,
              epochs=10,
              validation_data=val_dataset
              )

    model.save(model_name + ".keras")


def main():
    path_to_dataset = "../dataset_file"
    path_to_val_set = "../dataset_file"
    # generate the genre and validation datasets
    genre_dataset = tf.data.Dataset.load(path_to_dataset)
    val_dataset = tf.data.Dataset.load(path_to_val_set)
    model = build_model()

    # compile model with hyperparameters and RMSprop optimizer
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            learning_rate=0.0005,
            weight_decay=5e-4
            ),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="Accuracy")])
    save_name = "combined_2k6_32-64-128-256-512fiveconvlayer2kernavg_5e4LR5e4WD_es50epoch"
    os.mkdir("checkpoint/" + save_name)
    # Add checkpointing for model
    mod_check = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoint/" + save_name + "/checkpoint_{epoch}_{val_loss}_{loss}",
        monitor='val_Accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=True
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        mode="auto",
        restore_best_weights=True)

    csv_save = tf.keras.callbacks.CSVLogger("csv/" + save_name + ".csv")
    model.fit(genre_dataset,
              epochs=50,
              validation_data=val_dataset,
              callbacks=[mod_check, csv_save, early_stop]
              )
    model.save(save_name + ".keras")
    model_json = model.to_json()
    with open(save_name + ".json", "w") as output_file:
        json.dump(model_json, output_file)


if __name__ == "__main__":
    main()
