#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st


# In[2]:


st.title('Documentation')


# In[3]:


st.subheader("Homepage")
st.write("The homepage displays the dataset choosen and the evaluation done to compare between 2 different term weighting techniques.") 
st.write("The term weighting techbniques that we are comparing are Term Frequency-Inverse Documnet Frequency (TF-IDF) and Information Gain (IG).") 
st.write("We can choose the dataset from the drop down selection at the top of the page.")
st.write("There is also a input bar where the user can input their own review to check")
st.write("Then the model will predict the review and the result will be displayed as Positive or Negative Review")


# In[ ]:


st.subheader("Exploratory Data Analysis")
st.write("The EDA page displays the exploratory data analysis done on the selected dataset.")
st.write("There is a pie chart to show the percentage of positve and negative reviews.")
st.write("There are also a few WordClouds to show the common words in the negative and positive reviews")
st.write("We can choose the dataset from the drop down selection at the top of the page.")

