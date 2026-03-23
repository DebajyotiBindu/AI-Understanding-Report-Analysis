import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import nltk

TRAIN_PATH=r"D:\mlproject15\data\train\train.csv"

class preprocessing:
    def __init__(self,path):
        self.path=path

    def concatenation(self):
        data=pd.read_csv(self.path)
        data["keywords"] = data["keywords"].fillna("")
        data["transcription"] = data["transcription"].fillna("")
        data["text"]=data["keywords"]+" "+data["transcription"]
        stop_words=set(nltk.corpus.stopwords.words('english'))
        negation_words={'no','not', 'none', 'neither', 'never', 'without', 'without a', 'without any', 'without either', 'without neither', 'without no', 'without none', 'without one', 'without the', 'without them', 'without this', 'without them'}
        stop_words=stop_words-negation_words
        admin_noise = {'chief', 'complaint', 'history', 'present', 'illness', 'patient', 'physical', 'exam', 'examination', 'note', 'ordered', 'results'}
        stop_words.update(admin_noise)
        data['text']=data['text'].astype(str).str.lower()
        data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]','',x))
        data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        data.drop(['keywords','transcription'],axis=1,inplace=True)
        return data
    
    def split(self):
        data=self.concatenation()
        X_train=data.drop(['medical_specialty'],axis=1)
        y_train=data['medical_specialty']
        return X_train,y_train
    
    def tokenize(self):
        X_train_df,y_train=self.split()
        num_word=10000
        tokenizer=Tokenizer(num_words=num_word,oov_token="<OOV>")

        X_train=X_train_df['text'].astype(str)

        tokenizer.fit_on_texts(X_train) 
        os.makedirs('MODELS',exist_ok=True)
        joblib.dump(tokenizer,"MODELS/tokenizer.joblib")

        X_train_seq=tokenizer.texts_to_sequences(X_train)

        max_len=200
        X_train_pad=pad_sequences(X_train_seq,maxlen=max_len,padding='pre',truncating='pre')

        le=LabelEncoder()
        y_train_final=le.fit_transform(y_train)
        joblib.dump(le,"MODELS/labelencoder.joblib")

        print("Preprocessing done successfully")

        return X_train_pad,y_train_final 

if __name__=="__main__":
    obj=preprocessing(TRAIN_PATH)
    obj.tokenize()