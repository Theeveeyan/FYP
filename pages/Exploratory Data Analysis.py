#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
import streamlit as st


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
wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(str(mostcommon))
figure2 = plt.figure(figsize=(30,10), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Top 100 Most Common Words', fontsize=100)
plt.tight_layout(pad=0)
plt.show()


# In[17]:


fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.markdown("### Reviews")
    st.pyplot(figure1)
    
with fig_col2:
    st.markdown("### Common Words")
    st.pyplot(figure2)


# In[22]:


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
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(most_positive_words))
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(most_negative_words))

# Plot the word clouds
figure3 = plt.figure(figsize=(30, 10))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Top 100 Positive Words", fontsize=70)

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Top 100 Negative Words", fontsize=70)

plt.show()


# In[23]:


st.pyplot(figure3)


# In[ ]:




