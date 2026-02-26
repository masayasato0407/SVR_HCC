import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import gdown
import sksurv
from sksurv.ensemble import RandomSurvivalForest

class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "sklearn.tree._tree" and name == "Tree":
            from sklearn.tree._tree import Tree
            return Tree
        return super().find_class(module, name)

@st.cache_resource 
def load_model():
    destination = 'smartmodel.sav'
    if os.path.exists(destination):
        os.remove(destination)
        
    file_id = st.secrets["DRIVE_FILE_ID"]
    url = f'https://drive.google.com/uc?id={file_id}'
    
    with st.spinner('Downloading large model via gdown... Please wait.'):
        gdown.download(url, destination, quiet=False)
    
    try:
        with open(destination, 'rb') as f:
            model = CompatibilityUnpickler(f).load()
        if os.path.exists(destination):
            os.remove(destination)
        return model
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        raise e

try:
    rsf = load_model()
except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()

st.title('Prediction model for post-SVR HCC (SMART model)') 

with st.form('user_inputs'): 
    age = st.number_input('age (year)', min_value=18, max_value=100, value=60) 
    height = st.number_input('height (cm)', min_value=100.0, max_value=250.0, value=160.0) 
    weight = st.number_input('body weight (kg)', min_value=20.0, max_value=200.0, value=60.0)      
    PLT = st.number_input('Platelet count (×10^4/µL)', min_value=1.0, max_value=75.0, value=15.0)
    AFP = st.number_input('AFP (ng/mL)', min_value=0.1, max_value=100.0, value=5.0) 
    ALB = st.number_input('Albumin (g/dL)', min_value=1.0, max_value=7.0, value=4.0) 
    AST = st.number_input('AST (IU/L)', min_value=1, max_value=300, value=30)
    GGT = st.number_input('γ-GTP (IU/L)', min_value=1, max_value=1000, value=30)
    submitted = st.form_submit_button('Predict')

if submitted:
    BMI = weight / ((height/100)**2)
    X = pd.DataFrame(data={
        'age': [float(age)], 'BMI': [float(BMI)], 'PLT': [float(PLT)], 
        'AFP': [float(AFP)], 'ALB': [float(ALB)], 'AST': [float(AST)], 'GGT': [float(GGT)]
    }).astype(np.float32)
    
    surv_funcs = rsf.predict_survival_function(X)
    times = rsf.unique_times_
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for fn in surv_funcs:
        ax.step(fn.x, 1.0 - fn.y, where="post")
    
    ax.set_xlim(0, 10); ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted HCC incidence"); ax.set_xlabel("Years after SVR"); ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
    
    y_pred = rsf.predict_survival_function(X, return_array=True)[0]
    incidence = (1.0 - y_pred) * 100
    
    def get_val(year):
        idx = (np.abs(times - year)).argmin()
        return round(incidence[idx], 3)

    v1, v3, v5 = get_val(1), get_val(3), get_val(5)
    col1, col2, col3 = st.columns(3)
    col1.metric("1-year Risk", f"{v1}%")
    col2.metric("3-year Risk", f"{v3}%")
    col3.metric("5-year Risk", f"{v5}%")

