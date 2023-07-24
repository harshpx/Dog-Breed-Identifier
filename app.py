import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import requests
from io import BytesIO
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array



@st.cache_resource
def load_model():
    model = keras.models.load_model('models/model.keras')
    return model

def predictor(img):
    model = load_model()
    label_mapper = ['Afghan', 'African Wild Dog', 'Airedale', 'American Hairless',
       'American Spaniel', 'Basenji', 'Basset', 'Beagle',
       'Bearded Collie', 'Bermaise', 'Bichon Frise', 'Blenheim',
       'Bloodhound', 'Bluetick', 'Border Collie', 'Borzoi',
       'Boston Terrier', 'Boxer', 'Bull Mastiff', 'Bull Terrier',
       'Bulldog', 'Cairn', 'Chihuahua', 'Chinese Crested', 'Chow',
       'Clumber', 'Cockapoo', 'Cocker', 'Collie', 'Corgi', 'Coyote',
       'Dalmation', 'Dhole', 'Dingo', 'Doberman', 'Elk Hound',
       'French Bulldog', 'German Sheperd', 'Golden Retriever',
       'Great Dane', 'Great Perenees', 'Greyhound', 'Groenendael',
       'Irish Spaniel', 'Irish Wolfhound', 'Japanese Spaniel', 'Komondor',
       'Labradoodle', 'Labrador', 'Lhasa', 'Malinois', 'Maltese',
       'Mex Hairless', 'Newfoundland', 'Pekinese', 'Pit Bull',
       'Pomeranian', 'Poodle', 'Pug', 'Rhodesian', 'Rottweiler',
       'Saint Bernard', 'Schnauzer', 'Scotch Terrier', 'Shar_Pei',
       'Shiba Inu', 'Shih-Tzu', 'Siberian Husky', 'Vizsla', 'Yorkie']
    
    arr = img_to_array(img)
    arr = arr/255.0
    arr = np.expand_dims(arr,0)
    res = model.predict(arr)
    idx = res.argmax()
    val = label_mapper[idx]
    prob = res[0][idx]
    return val,prob

def main():
    st.title("Dog Bread Identifier")
    st.sidebar.markdown("<p>Made with ❤️ by Harsh Priye</p>", unsafe_allow_html=True)
    st.text("Upload an image or provide an image URL.")

    st.sidebar.title("Source")
    image_source = st.sidebar.radio("Select Image Source", ("Upload", "URL"))

    if image_source == "Upload":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            main_img = load_img(uploaded_file)
            image = load_img(uploaded_file, target_size=(224,224))
            st.image(main_img, caption="Uploaded Image", use_column_width=True)
            label, probability = predictor(image)
            st.write(f"Predicted Label: {label}\n\nProbability: {probability}")

    elif image_source == "URL":
        image_url = st.text_input("Enter Image URL")
        if st.button("Identify"):
            if image_url:
                response = requests.get(image_url)
                main_img = load_img(BytesIO(response.content))
                image = load_img(BytesIO(response.content), target_size=(224,224))
                st.image(main_img, caption="Uploaded Image", use_column_width=True)
                label, probability = predictor(image)
                st.write(f"Predicted Label: {label}\n\nProbability: {probability}")
            else:
                st.error("Please enter an image URL.")

if __name__ == "__main__":
    main()