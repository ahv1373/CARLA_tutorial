import os
from typing import Tuple

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from mock import Mock
import matplotlib.pyplot as plt

class TrainHyperParameters:
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3), number_of_classes: int = 2,
                 learning_rate: float = 0.001, batch_size: int = 32, number_of_epochs: int = 3) -> None:
        self.hyperparameters = Mock()
        self.hyperparameters.input_shape = input_shape
        self.hyperparameters.number_of_classes = number_of_classes
        self.hyperparameters.learning_rate = learning_rate
        self.hyperparameters.batch_size = batch_size
        self.hyperparameters.number_of_epochs = number_of_epochs


class TrainCustomCNN(TrainHyperParameters):
    def __init__(self, data_dir: str, checkpoint_dir: str = 'output/checkpoints') -> None:
        super().__init__()
        self.model = None

        np.random.seed(42)
        tf.random.set_seed(42)

        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir

    def form_data_generator(self) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')

        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.hyperparameters.input_shape[:2],
            batch_size=self.hyperparameters.batch_size,
            class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.hyperparameters.input_shape[:2],
            batch_size=self.hyperparameters.batch_size,
            class_mode='categorical')
        return train_generator, test_generator

    def model_builder(self):
        # Define the model architecture
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.hyperparameters.input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.hyperparameters.number_of_classes, activation='softmax')
        ])

    def train(self, train_generator, test_generator):
        optimizer = keras.optimizers.Adam(lr=self.hyperparameters.learning_rate)
        loss_fn = keras.losses.CategoricalCrossentropy()
        if self.model is not None:
            self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        else:
            raise ValueError("Model is not built.")

        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_dir, "model_checkpoint"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1)
        history = self.model.fit(
            train_generator,
            epochs=self.hyperparameters.number_of_epochs,
            validation_data=test_generator,
            callbacks=[checkpoint_callback])

        return history


if __name__ == "__main__":

    data_directory = '/home/ahv/PycharmProjects/Visual-Inertial-Odometry/simulation/CARLA/output/data_directory'
    checkpoint_directory = '/home/ahv/PycharmProjects/Visual-Inertial-Odometry/simulation/CARLA/output/checkpoint_directory"

    trainer = TrainCustomCNN(data_dir=data_directory, checkpoint_dir=checkpoint_directory)

    trainer.model_builder()

    train_generator, test_generator = trainer.form_data_generator()

    history = trainer.train(train_generator, test_generator)

    # Plot the training and validation accuracy over epochs
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
