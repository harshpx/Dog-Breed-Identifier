# Dog-Breed-Predictor-CNN-InceptionV3
Dog Breed Predictor Web App based on a Deep Learning classification model. created using **CNNs** and **InceptionV3** model (a 48 layers deep convolutional neural network).

**Tech Stack:** Python, Tensorflow, Streamlit

#### To run this project on your local system: 
Package Requirements: `TensorFlow`,`Numpy`,`Streamlit`

Install dependencies: `pip install tensorflow numpy streamlit`

To run the project:

```bash
mkdir Project
cd Project
git clone "[https://github.com/harshpx/Dog-Breed-Predictor-CNN-InceptionV3.git](https://github.com/harshpx/Dog-Breed-Predictor-CNN-InceptionV3.git)"
streamlit run app.py
```

in the terminal of file directory.

## Brief Project Description
* It is a Multiclass classification Model (70 Classes), that uses a medium sized image dataset consisting of approx. 9,000 images (on kaggle). These 70 classes are of 70 different breeds of dogs.
* This data is fed to a Large Convolutional Neural Network (with InceptionV3) as the base model.
* Model secured accuracy scores of **~91%** and **~92%** on Validation and Test data portions respectively.

## Training Notebook
[Kaggle: Dogs-Breed-Prediction](https://www.kaggle.com/code/harshpriye/dogs-breed-prediction-cnn-inceptionv3/notebook)
