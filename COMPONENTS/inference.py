import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
import numpy as np
import re 
import nltk 
import joblib 

TOKENIZER_PATH=r"D:\mlproject15\MODELS\tokenizer.joblib"
LE_PATH=r"D:\mlproject15\MODELS\labelencoder.joblib"
MODEL_PATH=r"D:\mlproject15\MODELS\model.keras"

class inference:
    def __init__(self):
        self.tokenizer=joblib.load(TOKENIZER_PATH)
        self.le=joblib.load(LE_PATH)
        self.model=load_model(MODEL_PATH)
        try:
            self.stop_words=set(nltk.corpus.stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words=set(nltk.corpus.stopwords.words('english'))
        
        negation_words={'no','not', 'none', 'neither', 'never', 'without', 'without a', 'without any', 'without either', 'without neither', 'without no', 'without none', 'without one', 'without the', 'without them', 'without this', 'without them'}
        self.stop_words=self.stop_words-negation_words
        admin_noise = {'chief', 'complaint', 'history', 'present', 'illness', 'patient', 'physical', 'exam', 'examination', 'note', 'ordered', 'results'}
        self.stop_words.update(admin_noise)
    
    def clean_text(self,text):
        text=text.lower()
        text=re.sub('[^a-zA-Z0-9\s]','',text)
        text=' '.join([word for word in text.split() if word not in (self.stop_words)])
        return text

    def prediction(self,text):
        cleaned_text=self.clean_text(text)
        seq_text=self.tokenizer.texts_to_sequences([cleaned_text])
        text_pad=pad_sequences(seq_text,maxlen=200,padding='post',truncating='post')

        predictions=[]
        y_preds=self.model.predict(text_pad)
        confidence=np.max(y_preds)*100
        y_preds_index=np.argmax(y_preds,axis=1)

        prediction=self.le.inverse_transform(y_preds_index)[0]

        y_probs=np.argsort(y_preds[0])[-5:]
        y_probs=y_probs[::-1]

        for i in y_probs:
            predictions.append(self.le.inverse_transform([i])[0])

        return prediction,confidence,predictions
    
if __name__=="__main__":
    obj=inference()
    text="HISTORY OF PRESENT ILLNESS: The patient is a 55-year-old male with a history of hypertension and tobacco use who presents with acute onset of sub-sternal chest pressure. The pain is described as crushing in nature, 10/10 in severity, and radiates to the left arm and jaw. Associated symptoms include diaphoresis and nausea. An EKG was performed in the field showing ST-segment elevation in leads V1-V4"

    prediction,confidence,predictions=obj.prediction(text)

    print("Result for the given sample is: ")
    print("Predicted disease class: ",prediction)
    print("Top 5 Predictions: ",predictions)
    print("Confidence score: ",confidence)
