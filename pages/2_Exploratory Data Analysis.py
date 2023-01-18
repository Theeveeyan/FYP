#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
import streamlit as st
import seaborn as sns
import plotly.express as px
from collections import Counter


# In[2]:


from wordcloud import WordCloud, ImageColorGenerator


# In[3]:


option = st.selectbox(
    'Which dataset would you like to choose?',
    ('Amazon', 'IMDb', 'Yelp'))

st.write('You selected:', option)


# In[4]:


if option == 'Amazon':
    filepath="https://raw.githubusercontent.com/Theeveeyan/FYP-Test/main/amazon_cells_labelled.txt"
elif option == 'IMDb':
    filepath="https://raw.githubusercontent.com/Theeveeyan/FYP-Test/main/imdb_labelled.txt"
elif option == 'Yelp':
    filepath="https://raw.githubusercontent.com/Theeveeyan/FYP-Test/main/yelp_labelled.txt"
amazon_df=pd.read_csv(filepath,
                        delimiter='\t',
                        header=None, 
                        names=['review', 'sentiment'])


# In[5]:


reviews = amazon_df["review"]
total_words = 0
total_reviews = 0
for review in reviews:
    words = review.split()
    total_words += len(words)
    total_reviews += 1
average_words = total_words / total_reviews
st.markdown("## Word Counts")

positive_reviews = amazon_df[amazon_df["sentiment"] == 1]
negative_reviews = amazon_df[amazon_df["sentiment"] == 0]

positive_words = 0
positive_reviews_count = 0
for review in positive_reviews["review"]:
    words = review.split()
    positive_words += len(words)
    positive_reviews_count += 1
positive_average_words = positive_words / positive_reviews_count

negative_words = 0
negative_reviews_count = 0
for review in negative_reviews["review"]:
    words = review.split()
    negative_words += len(words)
    negative_reviews_count += 1
negative_average_words = negative_words / negative_reviews_count

wc_avg = plt.figure(figsize=(15,8))
plt.bar(["Positive","Negative","Overall"], [positive_average_words,negative_average_words,average_words], color=['green','red','cyan'])
plt.xlabel("Sentiment")
plt.ylabel("Average Number of Words")
plt.title("AVG Number of Words per Review")
plt.show()

fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.pyplot(wc_avg)
    
with fig_col2:
    st.write("Average number of words in Overall dataset: {:.1f}".format(average_words))
    st.write("Average number of words in Positive reviews: {:.1f}".format(positive_average_words))
    st.write("Average number of words in Negative reviews: {:.1f}".format(negative_average_words))

amazon_df['review'] = amazon_df['review'].str.replace(r'[^\w\s]+', '')


# In[6]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[7]:


stop_words = stopwords.words('english')


# In[8]:


amazon_df['review'] = amazon_df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# In[9]:


amazon_df['review'] = amazon_df['review'].str.lower()


# In[10]:


amazon_df['review'] = amazon_df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# In[11]:


nltk.download('punkt');


# In[12]:


def tokenize(column):
    """Tokenizes a Pandas dataframe column and returns a list of tokens.

    Args:
        column: Pandas dataframe column (i.e. df['text']).

    Returns:
        tokens (list): Tokenized list, i.e. [Donald, Trump, tweets]
    """

    tokens = nltk.word_tokenize(column)
    return [w for w in tokens if w.isalpha()]


# In[13]:


amazon_df['review'] = amazon_df.apply(lambda x: tokenize(x['review']), axis=1)


# In[14]:


words = amazon_df['review']
allwords = []
for wordlist in words:
    allwords += wordlist
print(allwords)


# In[15]:


mostcommon = FreqDist(allwords).most_common(100)
wordcloud = WordCloud(width=1600, height=800, background_color='black', color_func=lambda *args, **kwargs: (0,255,255)).generate(str(mostcommon))
figure2 = plt.figure(figsize=(30,10), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word CLoud for Common Words', fontsize=100)
plt.tight_layout(pad=0)
plt.show()


# In[16]:


temp = amazon_df.groupby('sentiment').count()['review'].reset_index().sort_values(by='review',ascending=False)
temp.style.background_gradient(cmap='Purples')


# In[17]:


#amazon_df['temp_list'] = amazon_df['review'].apply(lambda x:str(x).split())
top = Counter([item for sublist in amazon_df['review'] for item in sublist])
temp1 = pd.DataFrame(top.most_common(20))
temp1.columns = ['Common_words','Count']
temp1.style.background_gradient(cmap='Blues')


# In[18]:


fig = px.bar(temp1, x="Count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 
             width=700, height=700)
fig.update_traces(marker_color='cyan')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()


# In[19]:


Positive_sent = amazon_df[amazon_df['sentiment']==1]
Negative_sent = amazon_df[amazon_df['sentiment']==0]
Positive_sent.head()


# In[20]:


#MosT common positive words
top = Counter([item for sublist in Positive_sent['review'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ['Positive_words','Count']
temp_positive.style.background_gradient(cmap='Greens')


# In[21]:


fig1 = px.bar(temp_positive, x="Count", y="Positive_words", title='Most Commmon Positive Words', orientation='h', 
             width=700, height=700)
fig1.update_traces(marker_color='green')
fig1.update_layout(yaxis={'categoryorder':'total ascending'})
fig1.show()


# In[22]:


top = Counter([item for sublist in Negative_sent['review'] for item in sublist])
temp_negative = pd.DataFrame(top.most_common(20))
temp_negative = temp_negative.iloc[1:,:]
temp_negative.columns = ['Negative_words','Count']
temp_negative.style.background_gradient(cmap='Reds')


# In[23]:


fig2 = px.bar(temp_negative, x="Count", y="Negative_words", title='Most Commmon Negative Words', orientation='h', 
             width=700, height=700)
fig2.update_traces(marker_color='red')
fig2.update_layout(yaxis={'categoryorder':'total ascending'})
fig2.show()


# In[24]:


words_pos = Positive_sent['review']
allwords_pos = []
for wordlist in words_pos:
    allwords_pos += wordlist
print(allwords_pos)


# In[25]:


mostcommon_pos = FreqDist(allwords_pos).most_common(50)
wordcloud = WordCloud(width=1600, height=800, background_color='black', color_func=lambda *args, **kwargs: (0,255,0)).generate(str(mostcommon_pos))
wc_pos = plt.figure(figsize=(30,10), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word Cloud for Positive Reviews', fontsize=100)
plt.tight_layout(pad=0)
plt.show()


# In[26]:


words_neg = Negative_sent['review']
allwords_neg = []
for wordlist in words_neg:
    allwords_neg += wordlist
print(allwords_neg)


# In[27]:


mostcommon_neg = FreqDist(allwords_neg).most_common(50)
wordcloud = WordCloud(width=1600, height=800, background_color='black', color_func=lambda *args, **kwargs: (255,0,0)).generate(str(mostcommon_neg))
wc_neg = plt.figure(figsize=(30,10), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word Cloud for Negative Reviews', fontsize=100)
plt.tight_layout(pad=0)
plt.show()


# In[28]:


st.markdown("### EDA on Overall dataset")
fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.write(fig)
    
with fig_col2:
    st.write(figure2)


# In[29]:


st.markdown("### EDA on Positive Reviews")
fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.write(fig1)
    
with fig_col2:
    st.write(wc_pos)


# In[30]:


st.markdown("### EDA on Negative Reviews")
fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.write(fig2)
    
with fig_col2:
    st.write(wc_neg)


# In[ ]:




