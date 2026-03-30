import os
from typing import Tuple

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from mock import Mock
from sklearn.metrics import classification_report, confusion_matrix

num_of_test_samples = 0
for root_dir, cur_dir, files in os.walk(r"D:\Set\test"):
    num_of_test_samples += len(files)
print('num_of_test_samples count:', num_of_test_samples)

class TrainHyperParameters:
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3), number_of_classes: int = 4,
                 learning_rate: float = 0.001, batch_size: int = 32, number_of_epochs: int = 5) -> None:
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

        # Set the seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir

    def form_data_generator(self) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')

        # Define the data generators for training and validation
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
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.hyperparameters.number_of_classes, activation='softmax')
        ])

    def train(self, train_generator, test_generator):
        # Define the optimizer and loss function
        optimizer = keras.optimizers.Adam(lr=self.hyperparameters.learning_rate)
        loss_fn = keras.losses.CategoricalCrossentropy()

        # Compile the model
        if self.model is not None:
            self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        else:
            raise ValueError('Model is not defined. Please call model_builder() first.')

        # Set up a checkpoint to save the best model weights
        if os.path.exists(self.checkpoint_dir) is False:
            os.makedirs(self.checkpoint_dir)
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.h5')

        checkpoint_cb = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                                        mode='max')

        # Train the model
        history = self.model.fit(train_generator,
                                 epochs=self.hyperparameters.number_of_epochs,
                                 validation_data=test_generator,
                                 callbacks=[checkpoint_cb])

        # Save the model architecture
        model_dir = os.path.join(self.checkpoint_dir, 'model')
        if os.path.exists(model_dir) is False:
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'model.json')
        model_json = self.model.to_json()
        with open(model_path, 'w') as json_file:
            json_file.write(model_json)
        # plot loss and accuracy on train and validation set
        self.plot_history(history)

        Y_predicton = self.model.predict_generator(test_generator, num_of_test_samples // self.hyperparameters.batch_size+1)
        y_prediction = np.argmax(Y_predicton, axis=1)
        print(confusion_matrix(test_generator.classes, y_prediction))
        target_names = ['ClearNoon', 'ClearSunset','fog','HardRainSunset']
        print(classification_report(test_generator.classes, y_prediction, target_names=target_names))



    def plot_history(self, history):
        matplotlib.use('Agg')
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.checkpoint_dir, 'loss_accuracy.png'), dpi=300)

    def exec(self):
        train_generator, test_generator = self.form_data_generator()
        self.model_builder()
        self.train(train_generator, test_generator)


if __name__ == '__main__':   
    data_dir_ = r"D:\Set"
    train_custom_cnn = TrainCustomCNN(data_dir_)
    train_custom_cnn.exec()
    
