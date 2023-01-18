#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st


# In[2]:


st.title('Documentation')


# In[3]:


st.subheader("Homepage")
st.write("The homepage displays the dataset choosen and the evaluation done to compare between 2 different term weighting techniques. The term weighting techbniques that we are comparing are Term Frequency-Inverse Documnet Frequency (TF-IDF) and Information Gain (IG). We can choose the dataset from the drop down selection at the top of the page.") 
st.write("There is also a input bar where the user can input their own review to check. Then the model will predict the review and the result will be displayed as Positive or Negative Review. The prediction is done by SVM Classifier and it uses TF-IDF as the feature selection method.")


# In[ ]:


st.subheader("Exploratory Data Analysis")
st.write("The EDA page displays the exploratory data analysis done on the selected dataset. We can choose the dataset we want using the dropdown selection. There are 3 selections to choose from, which are amazon, IMDb, and Yelp. There are several types of graph displayed. The first bar chart shows the number of words per review for the overall data and for the positive and negative reviews. "
        +"Then we have analysis done on common words. We have a bar chart which shows the most commonly used words in the overall dataset. The is also a word cloud to show the distribution of words. "
        +"As we scroll down we can see the same analysis but particular for the positive and negative reviews.")

