import pandas as pd
import numpy as np
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,Bidirectional,GRU,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import load_model
from COMPONENTS.preprocessing import preprocessing
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import re
import nltk
import joblib 

TOKENIZER_PATH=r"D:\mlproject15\MODELS\tokenizer.joblib"
LE_PATH=r"D:\mlproject15\MODELS\labelencoder.joblib"
TRAIN_PATH=r"D:\mlproject15\data\train\train.csv"
VAL_PATH=r"D:\mlproject15\data\val\val.csv"
TEST_PATH=r"D:\mlproject15\data\test\test.csv"
GLOVE_PATH=r"D:\mlproject15\archive\glove.6B.100d.txt"

class modelling:
    def __init__(self):
        pass

    def model(self,num_classes):
        input_dim=10000
        embedding_dim=100
        input_len=200
        embedding_matrix=self.embedding()
        model=Sequential(
            [
                Embedding(
                    input_dim=input_dim,
                    output_dim=embedding_dim,
                    input_length=input_len,
                    weights=[embedding_matrix],
                    trainable=False,
                    mask_zero=True
                ),

                Bidirectional(GRU(64,return_sequences=True)),
                Dropout(0.3),
                Bidirectional(GRU(32,return_sequences=False)),
                Dropout(0.3),
                Dense(64,activation='relu'),
                Dropout(0.3),
                Dense(num_classes,activation='softmax')
            ]
        )

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model 
    
    def val_tokenize(self):
        tokenizer=joblib.load(TOKENIZER_PATH)
        le=joblib.load(LE_PATH)

        data=pd.read_csv(VAL_PATH)
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

        val_x=data['text'].astype(str)
        val_y=data['medical_specialty']
        val_x=tokenizer.texts_to_sequences(val_x)
        val_x_padded=pad_sequences(val_x,maxlen=200,padding='pre',truncating='pre')

        val_y=le.transform(val_y)

        return val_x_padded,val_y
    
    def embedding(self):
        tokenizer=joblib.load(TOKENIZER_PATH)
        embedding_dim=100
        embeddings_index={}
        with open(GLOVE_PATH,encoding='utf-8') as f:
            for line in f:
                values=line.split()
                words=values[0]
                coef=np.asarray(values[1:],dtype='float32')
                embeddings_index[words]=coef

        num_words=10000
        embeddings_matrix=np.zeros((num_words,embedding_dim))

        for word,i in tokenizer.word_index.items():
            if i<num_words:
                embedding_vector=embeddings_index.get(word)
                if embedding_vector is not None:
                    embeddings_matrix[i]=embedding_vector
        
        return embeddings_matrix
    
    def train(self):
        val_x,val_y=self.val_tokenize()

        obj=preprocessing(TRAIN_PATH)
        X_train,y_train=obj.tokenize()
        le=joblib.load(LE_PATH)
        num_classes=len(le.classes_)

        model=self.model(num_classes)

        early_stop=EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )

        checkpoint=ModelCheckpoint(
            'MODELS/model.keras',
            monitor='val_loss',
            verbose=1,
            save_best_only=True 
        )

        lr_scheduler=ReduceLROnPlateau(
            monitor='val_loss',
            patience=3,
            verbose=1,
            min_delta=0.0001
        )

        class_weights=class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )

        dict_weights=dict(enumerate(class_weights))

        model.fit(
            X_train,y_train,
            epochs=20,
            validation_data=(val_x,val_y),
            batch_size=64,
            class_weight=dict_weights,
            callbacks=[early_stop,checkpoint,lr_scheduler]    
        )

        print("Model trained and saved successfully")

        model.layers[0].trainable=True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        new_early_stop=EarlyStopping(
            monitor='val_loss',
            patience=7,
            verbose=1,
            restore_best_weights=True
        )

        model.fit(
            X_train,y_train,
            epochs=30,
            validation_data=(val_x,val_y),
            batch_size=64,
            class_weight=dict_weights,
            callbacks=[new_early_stop,checkpoint,lr_scheduler]    
        )

        return model 
    
    def evaluation(self,model):
        tokenizer=joblib.load(TOKENIZER_PATH)
        le=joblib.load(LE_PATH)

        data=pd.read_csv(TEST_PATH)
        data["keywords"] = data["keywords"].fillna("")
        data["transcription"] = data["transcription"].fillna("")
        data["text"]=data["keywords"]+" "+data["transcription"]
        stop_words=set(nltk.corpus.stopwords.words('english'))
        negation_words={'no','not', 'none', 'neither', 'never', 'without', 'without a', 'without any', 'without either', 'without neither', 'without no', 'without none', 'without one', 'without the', 'without them', 'without this', 'without them'}
        stop_words=stop_words-negation_words
        data['text']=data['text'].astype(str).str.lower()
        data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]','',x))
        data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        data.drop(['keywords','transcription'],axis=1,inplace=True)

        test_x=data['text'].astype(str)
        test_y=data['medical_specialty']
        test_x=tokenizer.texts_to_sequences(test_x)
        test_x_padded=pad_sequences(test_x,maxlen=200,padding='post',truncating='post')

        test_y=le.transform(test_y)

        loss,accuracy=model.evaluate(
            test_x_padded,test_y
        )

        y_preds=model.predict(test_x_padded)
        y_preds=np.argmax(y_preds,axis=1)
        print(classification_report(test_y,y_preds))

        return test_x_padded,test_y,loss,accuracy
    
    def testing(self):
        model=load_model('MODELS/model.keras')
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        test_x_padded,test_y,loss,accuracy=self.evaluation(model)

        y_probs=model.predict(test_x_padded)
        top_5_preds=SparseTopKCategoricalAccuracy(k=5)
        top_5_preds.update_state(test_y,y_probs)
        top_acc=top_5_preds.result().numpy()

        print(f"Predictions: {top_acc:.4f}")
        print("Loss:",loss,"Accuracy:",accuracy)
    
if __name__=="__main__":
    obj=modelling()
    model=obj.train()

    model=load_model('MODELS/model.keras')
    _,_,loss,accuracy=obj.evaluation(model)

    obj.testing()
    print("Loss:",loss,"Accuracy:",accuracy)