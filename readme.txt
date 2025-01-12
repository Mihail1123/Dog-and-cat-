Image Classification Using VGG16 Model
This project implements image classification using the VGG16 model built with TensorFlow/Keras. The project consists of two main components:

Training the VGG16 Model on images.
Prediction using the trained model through a graphical user interface (GUI) implemented using the ipywidgets library.
Project Structure
The project consists of the following components:

1. VGG16 Model Training
Description:

The VGG16 architecture is used for image classification.
The model is trained on data stored in a directory where images are divided into subfolders (e.g., "Dogs" and "Cats").
Input images are resized to 224x224 pixels, normalized, and used for training the model.
Key functions:

read_img(path): Reads images from the specified directory.
build_network(input_shape, num_classes): Builds the VGG16 model.
train_network(model, x_train, y_train, x_val, y_val, batch_size, epochs, save_path): Trains the model and saves the result.
Steps:

Organize your images into folders, for example:
bash
Копировать код
/path/to/dataset/
├── Dogs/
└── Cats/
Ensure that the images are in .jpg format.
Run the code to train the model.
2. Prediction using the Trained Model
Description:

Once the model is trained, it can be used for predicting the classes of new images through a GUI.
Users can upload an image via the interface, and the model will predict whether it belongs to the "Dog" or "Cat" class.
Key components:

prepare_image(img_path): Prepares an image for input into the model.
classify_image(change): Processes the uploaded image and displays the result.
File upload widget and result display using ipywidgets.
Steps:

Load the trained model by running:
python
Копировать код
model = tf.keras.models.load_model('vgg16_model.h5')
Launch the graphical interface with ipywidgets to upload images and receive predictions.
Installation and Dependencies
To run the project, you'll need to install several Python libraries. You can use the requirements.txt file to install all dependencies:

text

tensorflow
numpy
scikit-image
matplotlib
ipywidgets
Pillow
Install the dependencies using pip:

bash
Копировать код
pip install -r requirements.txt
For working with Jupyter Notebooks and ipywidgets, you'll also need to install:

bash

pip install notebook
Usage
Model Training:
Prepare a directory with images divided into subfolders.
Run the main model training script. The trained model will be saved as vgg16_model.h5.
Image Classification:
Launch a Jupyter Notebook and use the ipywidgets interface to upload an image and make predictions.
The model will classify the image as either "Dog" or "Cat" based on its content.
Example Usage
Upload data (e.g., images of dogs and cats) into corresponding folders.
Train the model by running the main() function in the training script.
After training, launch the Jupyter Notebook with the image classification interface.
Upload an image through the GUI and receive a prediction.
Notes
The model is trained on 2 classes: "Dog" and "Cat". You can adapt the project to support more classes by modifying the parameters.
The image size input to the model must be 224x224 pixels, as this is the standard size for the VGG16 model.
License
This project is licensed under the MIT License. See the LICENSE file for details.