#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import sksurv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import BreslowEstimator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import pickle


# In[ ]:


@st.cache (allow_output_mutation=True) 
def load_model():
    return pickle.load(open("smartmodel.sav", 'rb'))

rsf = load_model()


# In[ ]:


st.title('Prediction model for post-SVR HCC (SMART model)') 


# In[ ]:


st.markdown("Enter the following items to display the predicted HCC risk")


# In[ ]:


with st.form('user_inputs'): 
  age=st.number_input('age (year)', min_value=0) 
  height=st.number_input('height (cm)', min_value=100.0) 
  weight=st.number_input('body weight (kg)', min_value=10.0)     
  PLT=st.number_input('Platelet count (×10^4/µL)', min_value=0.0)
  AFP=st.number_input('AFP (ng/mL)', min_value=0.0) 
  ALB=st.number_input('Albumin (g/dL)', min_value=0.0) 
  AST=st.number_input('AST (IU/L)', min_value=0)
  GGT=st.number_input('γ-GTP (IU/L)', min_value=0)
  st.form_submit_button() 


# In[ ]:


height2=height*height


# In[ ]:


BMI0=weight/height2


# In[ ]:


BMI=BMI0*10000


# In[ ]:


surv = rsf.predict_survival_function(pd.DataFrame(
    data={'age': [age],
          'BMI': [BMI],
          'PLT': [PLT],
          'AFP': [AFP],
          'ALB': [ALB],
          'AST': [AST],
          'GGT': [GGT],
         }
), return_array=True)

for i, s in enumerate(surv):
    plt.step(rsf.event_times_, s, where="post", label=str(i))
plt.xlim(0,10)
plt.ylim(0,1)
plt.ylabel("predicted HCC development")
plt.xlabel("years")
plt.grid(True)

plt.gca().invert_yaxis()

plt.yticks([0.0, 0.2, 0.4,0.6,0.8,1.0],
            ['100%', '80%', '60%', '40%', '20%', '0%'])
plt.savefig("img.png")


# In[ ]:


X=pd.DataFrame(
    data={'age': [age],
          'BMI': [BMI],
          'PLT': [PLT],
          'AFP': [AFP],
          'ALB': [ALB],
          'AST': [AST],
          'GGT': [GGT],
         }
)


# In[ ]:


rfscore0=pd.Series(rsf.predict(X))


# In[ ]:


rfscore=float(rfscore0)


# In[ ]:


st.header("HCC risk for submitted patient")


# In[ ]:


st.image ("img.png")


# In[ ]:


if rfscore < 0.956: 
    st.subheader("Risk grouping for HCC in the original article: Low risk")
    st.markdown("HCC incidence in the low-risk group of the original study cohort: 0/1000 person-year (95%CI:0-2/1000)")
elif rfscore >= 3.20: 
    st.subheader("Risk grouping for HCC in the original article: High risk")
    st.markdown("HCC incidence in the high-risk group of the original study cohort: 15 /1000 person-year (95%CI: 11-22/1000)")
else:
    st.subheader("Risk grouping for HCC in the original article: Intermediate risk")
    st.markdown("HCC incidence in the intermediate-risk group of the original study cohort: 3/1000 person-year (95%CI:1-6/1000)")

