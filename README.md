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
## Deployment
Clone: ```git clone https://github.com/your-username/AURA-MedXAI.git```

Install: ```pip install -r requirements.txt```

Launch: ```streamlit run app.py```
