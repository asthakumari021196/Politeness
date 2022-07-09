import pandas as pd
from distutils.log import error
import re
import string
import numpy as np

import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk import word_tokenize
from nltk.corpus import stopwords
# from keras.preprocessing.sequence import pad_sequences

# from keras.preprocessing.text import Tokenizer

# Generate and plot a synthetic imbalanced classification dataset
# from imblearn.over_sampling import SMOTE 
from collections import Counter



# from imblearn.over_sampling import RandomOverSampler



# class for cleaning the tweets 
class DataCleaning():
    def data_cleaning(self,text):
    # Remove any hyperlinks
        sentence = re.sub(r'https?:\/\/\S+', '', text)
        # Removing the RT
        sentence = re.sub(r'RT[\s]+', '', sentence)
        # Remove any '#'
        sentence = re.sub(r'#', '', sentence)
        # Remove the '\n' string
        sentence = re.sub('\\n', ' ', sentence)
        # Removing the @mention
        sentence = re.sub(r'@[A-Za-z0-9]+', '', sentence)
        # Data Cleansing
        sentence = re.sub(r'[^\w\s]', '', sentence)
        # Removing numbers
        sentence = re.sub(r'[0-9]', '', sentence)

        return sentence



    def remove_punctuation(self,text):
        return text.translate(str.maketrans('', '', string.punctuation))


    def p_tag_id_generation(self,data):
        data['p_tag_id'] = data['p_tag'].map({'P_0':0,
                             'P_1':1,
                             'P_2':2,
                             'P_3':3,
                             'P_4':4,
                             'P_5':5,
                             'P_6':6,
                             'P_7':7,
                             'P_8':8,
                             'P_9':9,
                             np.nan:'NY'},
                             na_action=None)
        return data                     


    def get_politeness(self,politeness_score):
        if politeness_score >= 5:
            return int(1)
        else:
            return int(0)     

    def remove_stopwords(self,text):
        removed = []
        stop_words = list(stopwords.words("english"))
        tokens = word_tokenize(text)
        for i in range(len(tokens)):
            if tokens[i] not in stop_words:
                removed.append(tokens[i])
        return " ".join(removed)       

    def make_tokenizer(self,texts, len_voc):

        t = Tokenizer(num_words=len_voc)
        t.fit_on_texts(texts)
        return t    

    # def oversamping(self,text,polarity_class):
    #     oversample = SMOTE()
    #     print(text,polarity_class)
    #     X, y = oversample.fit_resample(text, polarity_class)
    #     counter = Counter(y)
    #     print(counter)

    # def make_embedding_matrix(self,embedding, tokenizer, len_voc):
    #     all_embs = np.stack(embedding.values())
    #     emb_mean,emb_std = all_embs.mean(), all_embs.std()
    #     embed_size = all_embs.shape[1]
    #     word_index = tokenizer.word_index
    #     embedding_matrix = np.random.normal(emb_mean, emb_std, (len_voc, embed_size))
        
    #     for word, i in word_index.items():
    #         if i >= len_voc:
    #             continue
    #         embedding_vector = embedding.get(word)
    #         if embedding_vector is not None: 
    #             embedding_matrix[i] = embedding_vector
        
    #     return embedding_matrix    

    # def get_coefs(self,word,*arr): 
    #     return word, np.asarray(arr, dtype='float32')     

    # def load_embedding(self,file):
    #     if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
    #         embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    #     else:
    #         embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
    #     return embeddings_index


    def data_preprocessing(self,df): 

        df['txt'] = df[['txt']].applymap(lambda x: self.data_cleaning(x))
        print('data cleaned')
        
        df['txt'] = df['txt'].apply(lambda x: self.remove_punctuation(x))

        df['txt'] = df['txt'].apply(lambda x: self.remove_stopwords(x))

        df = self.p_tag_id_generation(df)

        df ['polite'] = df ['p_tag_id'].apply(self.get_politeness)
        
        print('after preprocessing')
        print(df['polite'].value_counts())
        print(df.head())
 
        # len_voc = 40000
        # max_len = 60
        # tokenizer = self.make_tokenizer(df['txt'], len_voc)

        # text = tokenizer.texts_to_sequences(df['txt'])
        # print('Tokenization completed')
        
        # text = pad_sequences(text, maxlen=max_len, padding='post', truncating='post')
        # print('Padding completed')

        # glove = self.load_embedding('glove.840B.300d.txt')
        # embed_mat = self.make_embedding_matrix(glove, tokenizer, len_voc)
        # print('Embedding completed')

 
        # text_emb = embed_mat[text]
        # text_size, max_len, embed_size = text_emb.shape
        # text_emb_r = text_emb.reshape(text_size, max_len*embed_size)

        # print('Before oversampling')
        # counter = Counter(df['p_tag_id'])
        # print(counter)


        # smt = SMOTE()
        # X_smote, y_smote = smt.fit_resample(text_emb_r, df['p_tag_id'])
        # counter = Counter(y_smote)
        # print('After oversampling')
        # print(counter)

        # print('done')
        # return X_smote,y_smote
        return df

        


