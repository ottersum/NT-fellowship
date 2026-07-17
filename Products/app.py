import streamlit as st
import numpy as np
import os
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# PATH BASE (IMPORTANT FIX)
# ======================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# load all the artifacts
# ======================

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "psychmed_model.pkl"))
    drug_enc = joblib.load(os.path.join(BASE_DIR, "drug_encoder.pkl"))
    drug_list = joblib.load(os.path.join(BASE_DIR, "drug_list.pkl"))
    age_map = joblib.load(os.path.join(BASE_DIR, "age_map.pkl"))
    time_map = joblib.load(os.path.join(BASE_DIR, "time_map.pkl"))
    n_features = joblib.load(os.path.join(BASE_DIR, "n_features.pkl"))

    df = pd.read_csv(os.path.join(BASE_DIR, "df_PMEI_final.csv"))
    diag_drug_map_df = pd.read_csv(os.path.join(BASE_DIR, "diagnosis_drug_map.csv"))

    return model, drug_enc, drug_list, age_map, time_map, n_features, df, diag_drug_map_df


model, drug_enc, drug_list, age_map, time_map, n_features, df, diag_drug_map_df = load_artifacts()

# =====================================================
# Make diagnosis to drug validation layer
# =====================================================

diag_to_drugs = (
    diag_drug_map_df.groupby("diagnosis")["drug_name"]
    .apply(set)
    .to_dict()
)

# =====================================================
# concept list and load embedder
# =====================================================

benefit_concepts = [
    "improved sleep", "better sleep quality", "falling asleep easily",
    "reduced insomnia", "improved mood", "mood stabilization",
    "emotional stability", "feeling like myself again",
    "reduced anxiety", "anxiety improvement", "panic attacks reduced",
    "reduced stress", "depression improvement",
    "improved depressive symptoms", "increased energy",
    "more energy", "improved focus", "improved concentration",
    "better cognitive clarity", "reduced rumination",
    "reduced overthinking", "improved daily functioning",
    "better social functioning", "improved quality of life",
    "improved motivation", "greater confidence"
]

burden_concepts = [
    "weight gain", "increased appetite", "fatigue", "daytime sleepiness",
    "low energy", "dry mouth", "insomnia", "sleep disturbance",
    "difficulty sleeping", "sexual dysfunction", "reduced libido",
    "pain", "headache", "migraines", "brain fog",
    "difficulty concentrating", "memory problems", "emotional blunting",
    "emotional numbness", "anxiety worsening", "depression worsening",
    "panic attacks worsening", "agitation", "irritability",
    "restlessness", "nausea", "gastrointestinal distress",
    "diarrhea", "constipation", "dizziness", "tremors",
    "sweating", "rash", "withdrawal symptoms", "brain zaps",
    "addiction", "dependence", "suicidal thoughts",
    "loss of personality", "reduced quality of life"
]

@st.cache_resource
def load_pmei_model():
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    benefit_embeddings = sentence_model.encode(benefit_concepts)
    burden_embeddings = sentence_model.encode(burden_concepts)
    return sentence_model, benefit_embeddings, burden_embeddings

sentence_model, benefit_embeddings, burden_embeddings = load_pmei_model()

# ======================
# embedding model
# ======================

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
embedding_dim = 384

# ======================
# feature builder for ML later
# ======================

def build_patient_features(embedding, age_id, gender_id, condition_id, time_on_drug):
    return np.concatenate([
        embedding,
        np.array([age_id, gender_id, condition_id, time_on_drug])
    ])

# ======================
# constants
# ======================

psychiatric_conditions = [
    "ADHD", "Bipolar Disorder", "Depression", "Depression and Anxiety (both)",
    "General Anxiety", "Insomnia", "OCD", "PMDD", "PTSD",
    "Panic Disorder", "Schizophrenia", "Seasonal Affective Disorder"
]

gender_map = {"male": 0, "female": 1, "other": 2, "prefer not to say": 3}

# ======================
# UI
# ======================

st.title("PsychMED")
st.header("Patient PMEI calculator and predictor")
st.warning(
    "Please have in mind that this information should NOT be used as medical advice. "
    "This is a research project and results are not clinically validated."
)

tab1, tab2 = st.tabs([
    "Patient Profile to Medication Recommender",
    "Medication History (PMEI calculator)"
])

# ======================
# TAB 1
# ======================

with tab1:
    st.header("Section A: Patient Profile")

    gender = st.selectbox("Gender", list(gender_map.keys()))
    age = st.selectbox("Age Group", list(age_map.keys()))
    diagnosis = st.selectbox("Diagnosis", psychiatric_conditions)
    symptoms = st.text_area("Usual experience with medication / Symptoms")

    def get_embedding(text):
        if not text or text.strip() == "":
            return np.zeros(embedding_dim)
        return embedder.encode(text)

    embedding = get_embedding(symptoms)

    age_id = age_map.get(age, -1)
    gender_id = gender_map.get(gender, 3)
    condition_id = psychiatric_conditions.index(diagnosis)

    patient_base = build_patient_features(
        embedding,
        age_id,
        gender_id,
        condition_id,
        time_on_drug=0,
        has_review=1 if symptoms.strip() else 0
    )

    def recommend(patient_base, diagnosis):
        results = []
        allowed_drugs = diag_to_drugs.get(diagnosis, set())

        for drug in drug_list:
            if drug not in allowed_drugs:
                continue
            if drug not in drug_enc.classes_:
                continue

            drug_id = drug_enc.transform([drug])[0]
            x = np.concatenate([patient_base, [drug_id]]).reshape(1, -1)

            pred = model.predict(x)[0]
            results.append((drug, pred))

        results.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(results[:5], columns=["Drug", "Predicted PMEI"])

    st.divider()

    if st.button("Generate Recommendations"):
        st.warning(
            "These are research-only recommendations and not clinically validated."
        )

        recs = recommend(patient_base, diagnosis)
        st.subheader("Top Recommended Medications")
        st.dataframe(recs, use_container_width=True)

# ======================
# TAB 2
# ======================

def compute_pmei(review_text, base_pmei=5):
    if not review_text or review_text.strip() == "":
        return base_pmei, 0

    emb = sentence_model.encode([review_text])
    benefit_sim = cosine_similarity(emb, benefit_embeddings).max()
    burden_sim = cosine_similarity(emb, burden_embeddings).max()

    raw = benefit_sim - burden_sim
    text_adjustment = np.clip(raw * 10, -2, 2)
    final_pmei = np.clip(base_pmei + text_adjustment, 1, 10)

    return final_pmei, text_adjustment

with tab2:
    st.header("Section B: PMEI Calculator")

    num_drugs = st.number_input("Past medications", 0, 10, 0)

    history_rows = []

    for i in range(num_drugs):
        st.subheader(f"Medication {i+1}")

        drug = st.selectbox(
            "Drug",
            df["drug_name"].dropna().unique(),
            key=f"drug_{i}"
        )

        base_pmei = st.slider("Base ratings", 2, 8, 5, key=f"base_{i}")
        review_text = st.text_area("Patient review text", key=f"text_{i}")

        final_pmei, adjustment = compute_pmei(review_text, base_pmei)

        history_rows.append({
            "drug": drug,
            "base_pmei": base_pmei,
            "adjustment": adjustment,
            "final_pmei": final_pmei
        })

    if history_rows:
        st.dataframe(pd.DataFrame(history_rows))
