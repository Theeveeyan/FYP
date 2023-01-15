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


figure1 = plt.figure(figsize=(30,10))
plt.title('Percentage of Reviews', fontsize=20)
amazon_df.sentiment.value_counts().plot(kind='pie', labels=['Positive', 'Negative'],
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})


# In[15]:


words = amazon_df['review']
allwords = []
for wordlist in words:
    allwords += wordlist
print(allwords)


# In[16]:


mostcommon = FreqDist(allwords).most_common(100)
wordcloud = WordCloud(width=1600, height=800, background_color='black', color_func=lambda *args, **kwargs: (0,255,255)).generate(str(mostcommon))
figure2 = plt.figure(figsize=(30,10), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Top 100 Most Common Words', fontsize=100)
plt.tight_layout(pad=0)
plt.show()


# In[18]:


# Import the necessary libraries
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Select the reviews and sentiments from the dataframe
reviews = amazon_df['review']
sentiments = amazon_df['sentiment']

# Split the reviews and sentiments into positive and negative sets
positive_reviews = [review for review, sentiment in zip(reviews, sentiments) if sentiment == 1]
negative_reviews = [review for review, sentiment in zip(reviews, sentiments) if sentiment == 0]

# Flatten the list of reviews into a list of words
positive_words = [word for review in positive_reviews for word in review]
negative_words = [word for review in negative_reviews for word in review]

# Count the frequency of each word in the positive and negative sets
positive_word_freq = Counter(positive_words)
negative_word_freq = Counter(negative_words)

# Sort the word frequencies in descending order
sorted_positive_word_freq = sorted(positive_word_freq.items(), key=lambda x: x[1], reverse=True)
sorted_negative_word_freq = sorted(negative_word_freq.items(), key=lambda x: x[1], reverse=True)

# Select the 100 most positive and negative words
most_positive_words = [word[0] for word in sorted_positive_word_freq[:100]]
most_negative_words = [word[0] for word in sorted_negative_word_freq[:100]]

# Generate the word clouds
positive_wordcloud = WordCloud(width=800, height=400, background_color='black', color_func=lambda *args, **kwargs: (0,255,0)).generate(' '.join(most_positive_words))
negative_wordcloud = WordCloud(width=800, height=400, background_color='black', color_func=lambda *args, **kwargs: (255,0,0)).generate(' '.join(most_negative_words))

# Plot the word clouds
figure3 = plt.figure(figsize=(30, 10))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Top 100 Positive Words", fontsize=70)

figure4 = plt.figure(figsize=(30, 10))
plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Top 100 Negative Words", fontsize=70)

plt.show()


# In[19]:


temp = amazon_df.groupby('sentiment').count()['review'].reset_index().sort_values(by='review',ascending=False)
temp.style.background_gradient(cmap='Purples')


# In[20]:


#amazon_df['temp_list'] = amazon_df['review'].apply(lambda x:str(x).split())
top = Counter([item for sublist in amazon_df['review'] for item in sublist])
temp1 = pd.DataFrame(top.most_common(20))
temp1.columns = ['Common_words','count']
temp1.style.background_gradient(cmap='Blues')


# In[21]:


fig = px.bar(temp1, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 
             width=700, height=700)
fig.update_traces(marker_color='cyan')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()


# In[22]:


Positive_sent = amazon_df[amazon_df['sentiment']==1]
Negative_sent = amazon_df[amazon_df['sentiment']==0]
Positive_sent.head()


# In[23]:


#MosT common positive words
top = Counter([item for sublist in Positive_sent['review'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ['Common_words','count']
temp_positive.style.background_gradient(cmap='Greens')


# In[24]:


fig1 = px.bar(temp_positive, x="count", y="Common_words", title='Most Commmon Positive Words', orientation='h', 
             width=700, height=700)
fig1.update_traces(marker_color='green')
fig1.update_layout(yaxis={'categoryorder':'total ascending'})
fig1.show()


# In[25]:


top = Counter([item for sublist in Negative_sent['review'] for item in sublist])
temp_negative = pd.DataFrame(top.most_common(20))
temp_negative = temp_negative.iloc[1:,:]
temp_negative.columns = ['Common_words','count']
temp_negative.style.background_gradient(cmap='Reds')


# In[26]:


fig2 = px.bar(temp_negative, x="count", y="Common_words", title='Most Commmon Negative Words', orientation='h', 
             width=700, height=700)
fig2.update_traces(marker_color='red')
fig2.update_layout(yaxis={'categoryorder':'total ascending'})
fig2.show()


# In[28]:


fig_col1, fig_col2, fig_col3 = st.columns(3)
with fig_col1:
#    st.markdown("### Common")
    st.write(fig)
    
with fig_col2:
#    st.markdown("### Positive Words")
    st.write(fig1)

with fig_col3:
#    st.markdown("### Negative Words")
    st.write(fig1)


# In[29]:


words_pos = Positive_sent['review']
allwords_pos = []
for wordlist in words_pos:
    allwords_pos += wordlist
print(allwords_pos)


# In[37]:


mostcommon = FreqDist(allwords_pos).most_common(50)
wordcloud = WordCloud(width=1600, height=800, background_color='black', color_func=lambda *args, **kwargs: (0,255,0)).generate(str(mostcommon))
wc_pos = plt.figure(figsize=(30,10), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word Cloud for Positive Words', fontsize=100)
plt.tight_layout(pad=0)
plt.show()


# In[35]:


words_neg = Negative_sent['review']
allwords_neg = []
for wordlist in words_neg:
    allwords_neg += wordlist
print(allwords_neg)


# In[36]:


mostcommon = FreqDist(allwords_neg).most_common(50)
wordcloud = WordCloud(width=1600, height=800, background_color='black', color_func=lambda *args, **kwargs: (255,0,0)).generate(str(mostcommon))
wc_neg = plt.figure(figsize=(30,10), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word Cloud for Negative Words', fontsize=100)
plt.tight_layout(pad=0)
plt.show()


# In[38]:


fig_col1, fig_col2, fig_col3 = st.columns(3)
with fig_col1:
#    st.markdown("### Common")
    st.write(figure2)
    
with fig_col2:
#    st.markdown("### Positive Words")
    st.write(wc_pos)

with fig_col3:
#    st.markdown("### Negative Words")
    st.write(wc_neg)


# In[ ]:




