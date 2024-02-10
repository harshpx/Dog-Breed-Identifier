# Dog-Breed-Identifier
Dog Breed Predictor Web App based on a Deep Learning classification model. created using **CNNs** and **InceptionV3** model (a 48 layers deep convolutional neural network).

**Tech Stack:** Python, Tensorflow, Streamlit

**This app is running on:** [https://dog-breed-identify.streamlit.app/](https://dog-breed-identify.streamlit.app/)

#### To run this project on your local system: 
Package Requirements: `TensorFlow`,`Numpy`,`Streamlit`

To run the project:

```bash
git clone "https://github.com/harshpx/Dog-Breed-Predictor-CNN-InceptionV3.git"

cd Dog-Breed-Predictor-CNN-InceptionV3

python3 -m venv venv 
or 
python3.11 -m venv venv

source ./venv/bin/activate

pip install numpy tensorflow streamlit

streamlit run app.py
```

in the terminal of file directory.

## Brief Project Description
* It is a Multiclass classification Model (70 Classes), that uses a medium sized image dataset consisting of approx. 9,000 images (on kaggle). These 70 classes are of 70 different breeds of dogs.
* This data is fed to a Large Convolutional Neural Network with Inception-V3 as the base model.
* Model secured accuracy scores of **~91%** and **~92%** on Validation and Test data portions respectively.

## Training Notebook
[Kaggle: Dog-Breed-Prediction](https://www.kaggle.com/code/harshpriye/dogs-breed-prediction-cnn-inceptionv3/notebook)
