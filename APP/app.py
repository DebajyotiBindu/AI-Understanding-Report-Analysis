import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib 
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from COMPONENTS.explain import explainer
from COMPONENTS.inference import inference
from COMPONENTS.modelling import modelling

@st.cache_resource
def essentials():
    model=load_model('MODELS/model.keras',compile=False)
    tokenizer=joblib.load('MODELS/tokenizer.joblib')
    le=joblib.load('MODELS/labelencoder.joblib')
    inf_obj=inference()
    exp_obj=explainer(tokenizer,le,model)
    return inf_obj,exp_obj

with st.sidebar:
    st.title("AURA (AI Understanding and Report Analysis)")
    st.info("This system classifies medical transcriptions into 40+ specialties with Explainable AI.")

st.title("Clinical Specialty Classifier & XAI")
st.write("Enter a medical transcription below to predict the department and see the AI's reasoning.")

input_text=st.text_area("Transcription",height=250,placeholder="Enter the medical transcription here---")
inf_obj,exp_obj=essentials()

if st.button("Analyze report"):
    if not input_text.strip():
        st.error("Please enter text to analyze.")
    
    else:
        with st.spinner("Diagnosing..."):
            preds,confidence,predictions=inf_obj.prediction(input_text)
        
        tab1, tab2 = st.tabs(["Diagnostic Results", "XAI Interpretation (LIME)"])

        with tab1:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(f"**Predicted Department:** {preds}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write("**Top 5 Possibilities:**")
                for i, p in enumerate(predictions, 1):
                    st.write(f"{i}. {p}")

            with col2:
                st.subheader("Top 5 Differential Diagnosis")
                df = pd.DataFrame({
                    "Specialty": predictions,
                    "Rank": [5, 4, 3, 2, 1] 
                })
                st.bar_chart(data=df, x="Specialty", y="Rank", color="#007bff")
        
        with tab2:
            st.subheader(f"Why did the AI pick {preds}?")
            st.write("The highlights below show which words influenced the decision.")
            with st.spinner("Generating explanation..."):
                explanation = exp_obj.explain(input_text)
                exp_html = explanation.as_html()
                
                custom_html = f"""
    <div style="background-color: white; color: black !important; padding: 25px; border-radius: 10px; font-family: sans-serif;">
        <style>
            /* 1. COMPLETELY WIPE OUT LIME HEADERS AND OVERLAPPING TITLES */
            .lime.header, .lime.title, h2, h3, 
            div[class*="headline"], /* Targets LIME internal headline divs */
            div[class*="label_title"] /* Target standard LIME label titles */ {{ 
                display: none !important; 
                visibility: hidden !important;
                opacity: 0 !important;
                height: 0px !important;
                margin: 0px !important;
                padding: 0px !important;
                position: absolute !important; /* Move it off canvas */
                top: -9999px !important;
            }} 
            
            /* 2. ADD CLEAN SPACING FOR THE VISUALIZATION */
            .lime.explanation {{ 
                margin-top: 30px !important; 
            }}

            /* 3. ENSURE BAR CHARTS HAVE PROPER POSITIONING (Not absolute) */
            .lime.labels, .lime.graph, .lime.table {{
                position: relative !important;
                top: 0px !important;
            }}
            
            /* 4. FORCE BLACK TEXT FOR READABILITY AGAINST WHITE BG */
            .lime.lexical, .lime.label, span, div, td, th {{ 
                color: black !important; 
            }}
        </style>
        {exp_html}
    </div>
"""
                components.html(custom_html, height=800, scrolling=True)