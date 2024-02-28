import pickle
import streamlit as st
# from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
# app=Flask(__name__)
regmodel=pickle.load(open('californiamodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

# @app.route('/')
# def homePage():
#     return 'Welcome ALL'


def predict(MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude):
    features_list=np.array([MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude])
    print(type(features_list))
    print(features_list.reshape(1,-1))
    transform_data=scaler.transform(np.array(features_list).reshape(1,-1))
    pred=regmodel.predict(transform_data)
    return pred

def main():
    st.title('California House Price Prediction')
    MedInc=st.text_input('MedInc','')
    HouseAge=st.text_input('HouseAge','')
    AveRooms=st.text_input('AveRooms','')
    AveBedrms=st.text_input('AveBedrms','')
    Population=st.text_input('Population','')
    AveOccup=st.text_input('AveOccup','')
    Latitude=st.text_input('Latitude','')
    Longitude=st.text_input('Longitude','')


    result=""
    if st.button('Predict'):
        result=predict(MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude)
    st.success('The output is {}'.format(result))
    if st.button('About'):
        st.text("Let's Learn")
        st.text('Built with StreamLit')

if __name__=='__main__':
    main()
