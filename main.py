import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

model = load_model("plant_disease.h5")

CLASS_LIST =  ['Corn-Common_rust','Potato-Early_blight','Tomato-Bacterial_spot']

st.title("Plant Disease Dectection App")
st.markdown("Upload image")


image = st.file_uploader("Choose Image",type="jpg")
submit = st.button('Predict')

if submit:
    if image is not None:
        file_bytes = np.asarray(bytearray(image.read()),dtype=np.int8)
        opencv = cv2.imdecode(file_bytes,1)

        st.image(opencv, channels="BGR")
        st.write(opencv.shape)

        opencv = cv2.resize(opencv,(256,256))
        opencv.shape = (1,256,256,3)
        Y_pred = model.predict(opencv)
        result = CLASS_LIST[np.argmax(Y_pred)]
        st.title("The leaf is"+result.split('-')[0]+" and has "+result.split('-')[1])
