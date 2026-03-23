# AURA: AI-driven Understanding & Report Analysis
*Explainable Clinical Specialty Classification for Modern Healthcare*

AURA is a specialized Deep Learning system designed to categorize unstructured medical transcriptions into 40+ clinical departments. Built for the 2027 batch technical portfolio, AURA solves the "Black Box" problem in healthcare AI by integrating LIME (Local Interpretable Model-agnostic Explanations) to provide real-time diagnostic reasoning.

## The Problem: Administrative Bias
Standard NLP models often "cheat" by identifying report headers like “CHIEF COMPLAINT” or “HISTORY OF PRESENT ILLNESS” to guess the specialty. AURA overcomes this through a Clinical-First Preprocessing Engine, forcing the model to ignore administrative noise and focus on high-density pathological tokens (e.g., ST-segment, T2-weighted, hypokinesis).

## Technical Benchmarks
* Top-1 Accuracy: 74.6%
* Top-5 Accuracy (Differential Diagnosis): 89.5%
* Inference Latency: <80ms (optimized for local GPU execution)
* Model: Bi-Directional GRU with 100d GloVe Embeddings.

## Key Engineering Decisions
* Symmetric Preprocessing: A custom pipeline that maintains medical negations (e.g., "no," "without") while stripping 50+ non-clinical keywords.
* Post-Sequence Padding: Strategically implemented to ensure the Bi-GRU processes critical clinical keywords at the start of a sequence without delay.
* XAI Heatmaps: Integrated LIME to generate lexical importance plots, providing a "visual audit trail" for clinicians.

## Folder Structure
```
AURA_MedXAI/
├── COMPONENTS/          # Modularized Engineering Logic
│   ├── inference.py     # Class-based inference engine
│   ├── explain.py       # LIME XAI implementation
│   └── modelling.py     # Bi-GRU architecture & training
├── MODELS/              # Serialized Artifacts (.keras, .joblib)
├── app.py               # Streamlit XAI Dashboard
└── requirements.txt     # Dependency Management
```
** Note:- In modelling.py you will find a Global path defined, named `GLOVE_PATH` That represents the path of the GLOVE embedding 100 Dimensions .txt file stored inside a folder named archive which due to size restraints iI didnot push here, so you have to manually create a folder named `archive` and store the .txt file inside it, which you may download from `glove.6B.100d.txt` - Kaggle as a zip folder which you can extract and run the modelling script to get the desired *.keras* inside MODELS (which you can create since i in this case made the MODELS folder manually so didnot add the os.makedirs(), so if you want you can add it or create a folder manually) and then you may run this scripts. If you want you can also download the GLOVE 200 dimensions .txt as well but if you do that you have to explicitely change the `embedding_dim` in my code from 100 to 200 and if you want to stick with 100 dimension as it is then you may proceed with the code as above. **

## Deployment
Clone: ```git clone https://github.com/your-username/AURA-MedXAI.git```

Install: ```pip install -r requirements.txt```

Launch: ```streamlit run app.py```
