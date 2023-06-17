import streamlit as st
import os
from fastai.vision.all import *

import pathlib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


path=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(path,'learn.pkl')
learn_inf =load_learner(model_path)
data_df = pd.read_excel('data2.xlsx', usecols=( 0,1,2,3,4), names=['user','user_id','location_id', 'location','score'])
location_matrix = data_df.pivot_table(index='user_id', columns='location', values='score')


st.title("Recommended location App")
st.write("Upload an image and the app will predict the corresponding label.")
#上传文件
uploaded_file=st.file_uploader("Choose an image...",type=["jpg","png","jpeg"])

# If the user has uploaded an image
if uploaded_file is not None:
    # Display the image
    image = PILImage.create(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Get the predicted label
    pred, pred_idx, probs = learn_inf.predict(image)
    
    if pred=="wudadao" :
        pred='五大道'
        st.write(pred)
    elif pred=="tianjinzhiyan" :
        pred='天津之眼'
        st.write(pred)
    elif pred=="shijizhong" :
        pred='世纪钟广场'
        st.write(pred)
    elif pred=="gulou":
        pred='鼓楼'
        st.write(pred)
    elif pred=="cifangzi":
        pred='瓷房子'
        st.write(pred)
    elif pred=="tianjinhuanlegu":
        pred='天津欢乐谷'
        st.write(pred)
    elif pred=="shuishanggongyuan":
        pred='水上公园'
        st.write(pred)
    elif pred=="tianjinmeishuguan":
        pred='天津美术馆'
        st.write(pred)
    elif pred=="guojiahaiyangbowuguan":
        pred='国家海洋博物馆'
        st.write(pred)
    else:
        pred='西开教堂'
        st.write(pred)
    st.title("The recommended location for you is：")
    def recommend_movies(pred, data_df=data_df, location_matrix=location_matrix):
        scores = pd.DataFrame(data_df.groupby('location')['score'].mean())
        scores['number_of_scores'] = data_df.groupby('location')['score']
        scores.sort_values('number_of_scores', ascending=False).head(10)
        AFO_user_score = location_matrix[pred]
        similar_to_air_force_one=location_matrix.corrwith(AFO_user_score)
        corr_AFO = pd.DataFrame(similar_to_air_force_one, columns=['correlation'])
        corr_AFO.dropna(inplace=True)
        result=corr_AFO.sort_values(by='correlation', ascending=False)
        similar_location_titles=result[1:4]
        return similar_location_titles
    st.write(recommend_movies(similar_location_titles))
