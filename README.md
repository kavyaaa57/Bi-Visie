# Bi-Visie
## Binary Image Recognition Project

This project is focused on building an image classification model using TensorFlow and Keras to differentiate between two classes: "Amala" and "Priya." The model is trained using a dataset of labeled images and is capable of predicting the class of a new image.

## Project Overview

- **Dataset**: The dataset is organized into training, validation, and test sets. The training and validation sets are used to train and evaluate the model, while the test set is used for final predictions.
- **Model Architecture**: The model is a Convolutional Neural Network (CNN) with four convolutional layers followed by max-pooling layers. The final layers consist of a flattening layer, a dense layer with 512 neurons, and an output layer with a sigmoid activation function for binary classification.
- **Training**: The model is trained using the `ImageDataGenerator` class for real-time data augmentation, with a learning rate of 0.001 and binary cross-entropy as the loss function.
- **Prediction**: The model can predict the class of new images by loading them from the test directory and displaying the results.

## Key Libraries Used

- TensorFlow
- Keras
- OpenCV
- Matplotlib
- NumPy

## Running the Project

1. Mount Google Drive to access the dataset.
2. Train the model using the provided training script.
3. Evaluate the model on the validation set.
4. Test the model on new images to classify them as either "Amala" or "Priya."

## File Structure

- `/train`: Contains training images, divided into subfolders for each class.
- `/validation`: Contains validation images, divided into subfolders for each class.
- `/test`: Contains test images to be classified by the trained model.

## Optimizers

- The model uses binary classification, so it is designed to distinguish between exactly two classes.
- The model is compiled using both `RMSprop` and `Adam` optimizers during experimentation, but the final implementation uses the `Adam` optimizer with categorical cross-entropy loss.


