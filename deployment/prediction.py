import pandas as pd
import numpy as np
import streamlit as st
# Library Load Model
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf

# Library Pre-Processing
import emoji
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import string
import tensorflow as tf
import tensorflow_hub as tf_hub
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import STOPWORDS

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


url = "https://www.kaggle.com/models/google/nnlm/TensorFlow2/tf2-preview-en-dim128-with-normalization/1"

hub_layer = tf_hub.KerasLayer(url, output_shape=[128], input_shape=[], dtype=tf.string)

#load model
# model_lstm_2 = load_model('model_lstm_2.h5',  custom_objects={'KerasLayer': hub_layer})
model_lstm_2 = tf.keras.models.load_model('model_lstm_2.h5', custom_objects={'KerasLayer': hub_layer})
# build text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs, mentions, and non-alphanumeric characters
    txt_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(txt_cleaning_re, ' ', text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace="")

    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in STOPWORDS])

    # Remove /r, /n characters, non-utf characters, numbers, and punctuations
    text = text.replace('\r', '').replace('\n', ' ')
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    text = re.sub('[0-9]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove additional stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])

    # Remove contractions
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # Clean hashtags
    text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text))
    text = " ".join(word.strip() for word in re.split('#|_', text))

    # Filter special characters
    text = " ".join(['' if ('$' in word) | ('&' in word) else word for word in text.split()])

    # Remove multiple spaces
    text = re.sub("\s\s+" , " ", text)

    return text
# Lemmatizing
lemmatizer = WordNetLemmatizer()
def lemmatizer_words(text):
    
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_text(text):
  
    text = clean_text(text)
    return text

def run():
    # membuat title
    st.title('Tweet Bullying Sentiment Detection')
    # st.subheader('Detecting Fake News')
    st.markdown('---')
    # Buat form
    with st.form(key='fake_news_detect'):
        st.write("## Tweet text")
        # URL input
        text = st.text_input("Enter text:")
        submitted = st.form_submit_button('Predict')
        if submitted:
                data_inf = {'text': text}
                data_inf = pd.DataFrame([data_inf])
                # Preprocess the text (apply the same preprocessing steps as used during training)
                data_inf['text'] = data_inf['text'].apply(lambda x: preprocess_text(x))
                data_inf = tokenizer.texts_to_sequences(data_inf)
                data_inf = pad_sequences(data_inf, maxlen=700)
                y_pred_inf = model_lstm_2.predict(data_inf)
                y_pred_inf = np.argmax(y_pred_lstm_2[0])

                # Display the prediction result
                if y_pred_inf == 0:
                    st.subheader("Bullying: Age")
                elif y_pred_inf == 1:
                    st.subheader("Bullying: Ethnicity")
                elif y_pred_inf == 2:
                    st.subheader("Bullying: Gender")
                elif y_pred_inf == 3:
                    st.subheader("Bullying: Not Cyberbullying")
                elif y_pred_inf == 4:
                    st.subheader("Bullying: Other Cyberbullying")
                else:
                    st.subheader("Prediction: Religion")

                st.subheader("Extracted Text:")
                st.write(text)

if __name__ == '__main__':
    run()