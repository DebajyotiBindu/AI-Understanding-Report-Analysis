import lime
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import nltk
import re 

TOKENIZER_PATH=r"D:\mlproject15\MODELS\tokenizer.joblib"
LE_PATH=r"D:\mlproject15\MODELS\labelencoder.joblib"
MODEL_PATH=r"D:\mlproject15\MODELS\model.keras"

class explainer:
    def __init__(self,tokenizer,label_encoder,model):
        self.model=model
        self.tokenizer=tokenizer 
        self.le=label_encoder
        self.text_explainer=LimeTextExplainer(class_names=list(self.le.classes_))
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
    
    def predict_prob(self,text):
        cleaned_text=[self.clean_text(t) for t in text]
        sequences=self.tokenizer.texts_to_sequences(cleaned_text)
        text_pad=pad_sequences(sequences,maxlen=200,padding='post',truncating='post')
        
        return self.model.predict(text_pad,verbose=1)
    
    def explain(self,text):
        num_feature=10
        result=self.predict_prob([text])
        top_class=np.argmax(result[0])

        explanation=self.text_explainer.explain_instance(
            text,
            self.predict_prob,
            num_features=num_feature,
            labels=[top_class],
            num_samples=500
        )

        return explanation

if __name__=="__main__":
    tokenizer=joblib.load(TOKENIZER_PATH)
    le=joblib.load(LE_PATH)
    model=load_model(MODEL_PATH)

    obj=explainer(tokenizer,le,model)
    text="The patient presents with acute sub-sternal chest pain and diaphoresis."
    exp=obj.explain(text)
    exp.save_to_file("LIME\exp.html")
    print("Explanation saved")