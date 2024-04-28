# Plant Disease Classifier with Deep Learning

This project implements a Plant Disease Classifier application using deep learning techniques using PlantVillage Dataset.

## Overview

The application uses a CNN model to detect for the possible disease the plant is suffereing from along with the confidence level of detection. The current model version is having 97.01% accuracy on the training set.It comprises three main modules:

1. **research.ipynb**: Jupyter Notebook where the model was trained

2. **models**: Place where all the versions of the trained model are saved

3. **main-tf-serving.py**: Contains the code for the fast api which uses tensorflow-serving to make predictions

## Instructions

1. **Setup**
   - Ensure necessary dependencies (TensorFlow, Docker, etc.) are installed.
   - Place dataset in the specified directory (`../Dataset`).
   - Ensure that you have the tensorflow-serving docker image pulled.
   - make setup for the tensorflow-serving using the models.config file and appropriate commands that you will find in comments in the main-tf-serving file

2. **Usage**
   - Run `uvicorn main-tf-serving:app --reload` inside the api folder to start FastAPI server.
   - Select a image and make a post request to the predict endpoint (`localhost:8000/predict`).

3. **Further Development**
   - Experiment with different model architectures or datasets for improved accuracy.
   - Model quantization and developing tf-lite model for prediction.
   - Enhance functionalities by creating a react-native mobile app for predication.
   - Uploading the predict function on GCP and using Bucket to store the saved model.

## Contributing
Currently I am working on model quantization and developing tf-lite model for prediction which would help this application to run on small devices having low computational ability and alongside working on React-Native frontend for the mobile application, Contributions, issues, and feature requests are welcome!
