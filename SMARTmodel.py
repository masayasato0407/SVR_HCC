import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sksurv
import os
from sksurv.ensemble import RandomSurvivalForest

# --- バージョン互換性のためのクラス ---
class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "sklearn.tree._tree" and name == "Tree":
            from sklearn.tree._tree import Tree
            return Tree
        return super().find_class(module, name)

@st.cache_resource 
def load_model():
    # ファイル名をここに記載（GitHub上の大文字小文字と完全に一致させてください）
    model_file = "smartmodel.sav"
    
    if not os.path.exists(model_file):
        st.error(f"Error: '{model_file}' not found in the repository.")
        st.stop()
        
    with open(model_file, 'rb') as f:
        try:
            return CompatibilityUnpickler(f).load()
        except Exception as e:
            st.error(f"Loading error: {e}")
            st.stop()

# モデル読み込み
rsf = load_model()

# --- UI部分 ---
st.title('Prediction model for post-SVR HCC (SMART model)') 
st.markdown("Enter the following items to display the predicted HCC risk")

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
        'age': [age], 'BMI': [BMI], 'PLT': [PLT], 'AFP': [AFP], 
        'ALB': [ALB], 'AST': [AST], 'GGT': [GGT]
    })
    
    # 予測
    surv_funcs = rsf.predict_survival_function(X)
    times = rsf.unique_times_
    
    # グラフ描画
    fig, ax = plt.subplots()
    for fn in surv_funcs:
        ax.step(fn.x, 1.0 - fn.y, where="post")
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted HCC incidence")
    ax.set_xlabel("Years after SVR")
    ax.grid(True)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    st.header("HCC risk for submitted patient")
    st.pyplot(fig)
    
    # 指標算出
    incidence = (1.0 - rsf.predict_survival_function(X, return_array=True)[0]) * 100
    df_merge = pd.DataFrame({'timepoint (year)': times, 'predicted HCC incidence (%)': incidence})
    
    def get_val(year):
        idx = (np.abs(times - year)).argmin()
        return round(df_merge.iloc[idx, 1], 3)

    one, three, five = get_val(1), get_val(3), get_val(5)

    st.subheader("Risk Assessment")
    if five < 1.33: 
        st.success(f"Risk grouping: **Low risk** (5-year risk: {five}%)")
    elif five >= 5.03: 
        st.error(f"Risk grouping: **High risk** (5-year risk: {five}%)")
    else:
        st.warning(f"Risk grouping: **Intermediate risk** (5-year risk: {five}%)")

    col1, col2, col3 = st.columns(3)
    col1.metric("1-year Risk", f"{one}%")
    col2.metric("3-year Risk", f"{three}%")
    col3.metric("5-year Risk", f"{five}%")
    
    st.dataframe(df_merge, height=400)
