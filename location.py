import streamlit as st
import os
from fastai.vision.all import *

import pathlib
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

path=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(path,'learn.pkl')
learn_inf =load_learner(model_path)

pathlib.PosixPath=temp

#上传文件
uploaded_file=st.file_uploader("Choose an image...",type=["jpg","png","jpeg"])

# If the user has uploaded an image
if uploaded_file is not None:
    # Display the image
    image = PILImage.create(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Get the predicted label
    pred, pred_idx, probs = learn_inf.predict(image)
    st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")