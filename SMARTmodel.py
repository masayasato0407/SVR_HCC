import streamlit as st
import pickle
import pd as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import sksurv
from sksurv.ensemble import RandomSurvivalForest

class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "sklearn.tree._tree" and name == "Tree":
            from sklearn.tree._tree import Tree
            return Tree
        return super().find_class(module, name)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

@st.cache_resource 
def load_model():
    file_id = st.secrets["DRIVE_FILE_ID"]
    destination = 'smartmodel.sav'
    if not os.path.exists(destination):
        download_file_from_google_drive(file_id, destination)
    
    with open(destination, 'rb') as f:
        model = CompatibilityUnpickler(f).load()
    
    # 読み込み完了後、ディスク上の重いファイルを削除して空き容量を確保
    if os.path.exists(destination):
        os.remove(destination)
    return model

try:
    rsf = load_model()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

st.title('Prediction model for post-SVR HCC (SMART model)') 
st.markdown("Enter values to display the predicted HCC risk")

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
    # メモリ節約のため float32 を使用
    X = pd.DataFrame(data={
        'age': [float(age)], 'BMI': [float(BMI)], 'PLT': [float(PLT)], 
        'AFP': [float(AFP)], 'ALB': [float(ALB)], 'AST': [float(AST)], 'GGT': [float(GGT)]
    }).astype(np.float32)
    
    surv_funcs = rsf.predict_survival_function(X)
    times = rsf.unique_times_
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for fn in surv_funcs:
        ax.step(fn.x, 1.0 - fn.y, where="post", color="red")
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted HCC incidence")
    ax.set_xlabel("Years after SVR")
    ax.grid(True, linestyle='--')
    
    st.header("HCC risk assessment")
    st.pyplot(fig)
    plt.close(fig) # メモリ解放
    
    # リスク数値の算出
    y_pred = rsf.predict_survival_function(X, return_array=True)[0]
    incidence = (1.0 - y_pred) * 100
    df_merge = pd.DataFrame({'timepoint (year)': times, 'incidence (%)': incidence})
    
    def get_val(year):
        idx = (np.abs(times - year)).argmin()
        return round(df_merge.iloc[idx, 1], 3)

    v1, v3, v5 = get_val(1), get_val(3), get_val(5)

    col1, col2, col3 = st.columns(3)
    col1.metric("1-year Risk", f"{v1}%")
    col2.metric("3-year Risk", f"{v3}%")
    col3.metric("5-year Risk", f"{v5}%")
    
    if v5 < 1.33:
        st.success(f"Risk Group: **Low Risk** (5yr: {v5}%)")
    elif v5 >= 5.03:
        st.error(f"Risk Group: **High Risk** (5yr: {v5}%)")
    else:
        st.warning(f"Risk Group: **Intermediate Risk** (5yr: {v5}%)")
