import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "/Users/lina/Desktop/NT-WebMD/Data/df_PMEI_final.csv"
    )

df = load_data()

# =====================================================
# LOAD MODELS (ENSEMBLE)
# =====================================================

@st.cache_resource
def load_models():

    ridge = joblib.load(
        "/Users/lina/Desktop/NT-WebMD/Data/streamlit/ridge_model.pkl"
    )

    histgb = joblib.load(
        "/Users/lina/Desktop/NT-WebMD/Data/streamlit/histgb_model.pkl"
    )

    mlp = joblib.load(
        "/Users/lina/Desktop/NT-WebMD/Data/streamlit/mlp_model.pkl"
    )

    drug_encoder = joblib.load(
        "/Users/lina/Desktop/NT-WebMD/Data/streamlit/drug_label_encoder.pkl"
    )

    condition_encoder = joblib.load(
        "/Users/lina/Desktop/NT-WebMD/Data/streamlit/condition_label_encoder.pkl"
    )

    gender_encoder = joblib.load(
        "/Users/lina/Desktop/NT-WebMD/Data/streamlit/gender_label_encoder.pkl"
    )

    age_map = joblib.load(
        "/Users/lina/Desktop/NT-WebMD/Data/streamlit/age_map.pkl"
    )

    return (
        ridge,
        histgb,
        mlp,
        drug_encoder,
        condition_encoder,
        gender_encoder,
        age_map
    )

(
    ridge_model,
    histgb_model,
    mlp_model,
    drug_encoder,
    condition_encoder,
    gender_encoder,
    AGE_MAP
) = load_models()


# =====================================================
# DIAGNOSIS LIST
# =====================================================

psychiatric_conditions = [
    "Depression",
    "Major Depressive Disorder",
    "Repeated Episodes of Anxiety",
    "Anxious",
    "Panic Disorder",
    "Bipolar Depression",
    "Bipolar I Disorder with Most Recent Episode Mixed",
    "Schizophrenia",
    "Additional Medications to Treat Depression",
    "Anxiousness associated with Depression",
    "Chronic Trouble Sleeping",
    "Mania associated with Bipolar Disorder",
    "Posttraumatic Stress Syndrome",
    "Bipolar Disorder in Remission",
    "Obsessive Compulsive Disorder"
]

# =====================================================
# DIAGNOSIS → DRUG CONSTRAINT MAP
# =====================================================

DIAGNOSIS_TO_DRUGS = {

    # =====================================================
    # DEPRESSION
    # =====================================================

    "Depression": [
        "Sertraline Oral",
        "Fluoxetine Oral",
        "Citalopram Oral",
        "Escitalopram Oxalate Oral",
        "Paroxetine Oral",
        "Paroxetine Mesylate Oral",
        "Venlafaxine Oral",
        "Venlafaxine Besylate Oral",
        "Duloxetine Oral",
        "Cymbalta Oral",
        "Bupropion HCl Oral",
        "Wellbutrin Oral",
        "Wellbutrin SR Oral",
        "Wellbutrin XL Oral",
        "Mirtazapine Oral",
        "Amitriptyline Oral",
        "Nortriptyline Oral",
        "Doxepin Oral"
    ],

    # =====================================================
    # MAJOR DEPRESSIVE DISORDER
    # =====================================================

    "Major Depressive Disorder": [
        "Sertraline Oral",
        "Fluoxetine Oral",
        "Citalopram Oral",
        "Escitalopram Oxalate Oral",
        "Paroxetine Oral",
        "Paroxetine Mesylate Oral",
        "Venlafaxine Oral",
        "Venlafaxine Besylate Oral",
        "Duloxetine Oral",
        "Cymbalta Oral",
        "Bupropion HCl Oral",
        "Wellbutrin Oral",
        "Wellbutrin SR Oral",
        "Wellbutrin XL Oral",
        "Mirtazapine Oral",
        "Amitriptyline Oral",
        "Nortriptyline Oral",
        "Doxepin Oral"
    ],

    # =====================================================
    # ANXIETY DISORDERS
    # =====================================================

    "Repeated Episodes of Anxiety": [
        "Sertraline Oral",
        "Paroxetine Oral",
        "Fluoxetine Oral",
        "Escitalopram Oxalate Oral",
        "Venlafaxine Oral"
    ],

    "Anxious": [
        "Sertraline Oral",
        "Hydroxyzine HCl Oral",
        "Hydroxyzine Pamoate Oral",
        "Diazepam Oral",
        "Lorazepam Oral"
    ],

    "Panic Disorder": [
        "Paroxetine Oral",
        "Sertraline Oral",
        "Fluoxetine Oral",
        "Clonazepam Oral",
        "Alprazolam Oral",
        "Lorazepam Oral",
        "Venlafaxine Oral"
    ],

    "Anxiousness associated with Depression": [
        "Sertraline Oral",
        "Escitalopram Oxalate Oral",
        "Paroxetine Oral",
        "Venlafaxine Oral",
        "Duloxetine Oral"
    ],

    # =====================================================
    # PTSD
    # =====================================================

    "Posttraumatic Stress Syndrome": [
        "Sertraline Oral",
        "Paroxetine Oral",
        "Fluoxetine Oral",
        "Venlafaxine Oral"
    ],

    # =====================================================
    # BIPOLAR DISORDERS
    # =====================================================

    "Bipolar Depression": [
        "Quetiapine Oral",
        "Seroquel Oral",
        "Seroquel XR Oral",
        "Lurasidone Oral",
        "Latuda Oral",
        "Lithium Carbonate Oral",
        "Lithobid Oral",
        "Lamotrigine Oral"
    ],

    "Bipolar I Disorder with Most Recent Episode Mixed": [
        "Lithium Carbonate Oral",
        "Lithobid Oral",
        "Valproic Acid Oral",
        "Divalproex Oral",
        "Quetiapine Oral",
        "Seroquel Oral",
        "Risperidone Oral",
        "Olanzapine Oral",
        "Carbamazepine Oral"
    ],

    "Mania associated with Bipolar Disorder": [
        "Lithium Carbonate Oral",
        "Lithobid Oral",
        "Valproic Acid Oral",
        "Divalproex Oral",
        "Carbamazepine Oral",
        "Olanzapine Oral",
        "Risperidone Oral",
        "Haloperidol Oral"
    ],

    "Bipolar Disorder in Remission": [
        "Lithium Carbonate Oral",
        "Lithobid Oral",
        "Lamotrigine Oral",
        "Divalproex Oral",
        "Carbamazepine Oral"
    ],

    # =====================================================
    # SCHIZOPHRENIA
    # =====================================================

    "Schizophrenia": [
        "Risperidone Oral",
        "Risperdal Oral",
        "Olanzapine Oral",
        "Zyprexa Oral",
        "Quetiapine Oral",
        "Seroquel Oral",
        "Aripiprazole Oral",
        "Abilify Oral",
        "Paliperidone Oral",
        "Invega Oral",
        "Ziprasidone Oral",
        "Geodon Oral",
        "Lurasidone Oral",
        "Latuda Oral",
        "Haloperidol Oral",
        "Chlorpromazine Oral",
        "Fluphenazine Oral"
    ],

    # =====================================================
    # OCD
    # =====================================================

    "Obsessive Compulsive Disorder": [
        "Fluoxetine Oral",
        "Sertraline Oral",
        "Paroxetine Oral",
        "Fluvoxamine Oral",
        "Clomipramine Oral"
    ],

    # =====================================================
    # SLEEP DISORDERS
    # =====================================================

    "Chronic Trouble Sleeping": [
        "Trazodone Oral",
        "Zolpidem Oral",
        "Doxepin Oral",
        "Hydroxyzine HCl Oral",
        "Hydroxyzine Pamoate Oral"
    ]
}

# =====================================================
# EMBEDDING MODEL
# =====================================================

def embed(text):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(text)

# =====================================================
# DRUG ENCODING
# =====================================================

def get_drug_id(drug_name):

    try:
        return drug_encoder.transform([drug_name])[0]
    except:
        return -1


def get_condition_id(condition):

    try:
        return condition_encoder.transform([condition])[0]
    except:
        return -1


def get_gender_id(gender):

    try:
        return gender_encoder.transform([gender])[0]
    except:
        return -1


def get_age_id(age_group):

    return AGE_MAP.get(age_group, -1)

# =====================================================
# PROFILE BUILDER
# =====================================================

def build_review_text(drug_entries):

    reviews = []

    for d in drug_entries:

        notes = d["notes"]

        if notes.strip() == "":
            notes = "[NO REVIEW PROVIDED]"

        reviews.append(notes)

    return " ".join(reviews)

# =====================================================
# MODEL PREDICTION
# =====================================================

def predict_ptbi_ensemble(
    review_vec,
    drug_name,
    diagnosis,
    gender,
    age_group,
    rating,
    has_review
):

    drug_id = get_drug_id(drug_name)

    condition_id = get_condition_id(
        diagnosis
    )

    gender_id = get_gender_id(
        gender
    )

    age_id = get_age_id(
        age_group
    )

    structured = np.array([
        rating,
        age_id,
        drug_id,
        condition_id,
        gender_id,
        has_review
    ])

    X = np.concatenate([
        review_vec,
        structured
    ]).reshape(1, -1)

    preds = np.array([

        ridge_model.predict(X)[0],

        histgb_model.predict(X)[0],

        mlp_model.predict(X)[0]

    ])

    preds = np.clip(preds, 0, 1)

    return (
        preds.mean(),
        preds.std(),
        preds
    )

# =====================================================
# DRUG SCORING
# =====================================================

def compute_drug_ptbi(
    review_vec,
    drugs,
    diagnosis,
    gender,
    age_group,
    rating,
    has_review
):

    results = []

    for drug in drugs:

        mean, std, _ = predict_ptbi_ensemble(
            review_vec,
            drug,
            diagnosis,
            gender,
            age_group,
            rating,
            has_review
        )

        results.append({
            "drug": drug,
            "predicted_PTBI": mean,
            "uncertainty": std,
            "lower_bound": mean - std,
            "upper_bound": mean + std
        })

    return (
        pd.DataFrame(results)
        .sort_values(
            "predicted_PTBI"
        )
    )

# =====================================================
# RECOMMENDATION (CONSTRAINED)
# =====================================================

def recommend_drugs(
    review_vec,
    diagnosis,
    gender,
    age_group,
    rating,
    has_review
):

    allowed_drugs = DIAGNOSIS_TO_DRUGS.get(
        diagnosis,
        []
    )

    if len(allowed_drugs) == 0:

        allowed_drugs = (
            df["drug_name"]
            .dropna()
            .unique()
            .tolist()
        )

    return compute_drug_ptbi(
        review_vec,
        allowed_drugs,
        diagnosis,
        gender,
        age_group,
        rating,
        has_review
    ).head(5)

# =====================================================
# LOAD APP
# =====================================================

st.title("PsychMED")

st.write(
    "PsychMED estimates individualized PTBI scores using an ensemble model "
    "and evaluates medication burden based on patient profiles."
)

# =====================================================
# PATIENT INPUT
# =====================================================

st.header("Patient Information")

gender = st.selectbox(
    "Gender",
    ["male", "female", "other", "prefer not to say"]
)

age = st.selectbox(
    "Age Group",
    [
        "13-18",
        "19-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65-74",
        "75 or over"
    ]
)

diagnosis = st.selectbox(
    "Diagnosis",
    psychiatric_conditions
)

symptoms = st.text_area(
    "Symptoms",
    placeholder="e.g. fatigue, insomnia, panic attacks..."
)

# =====================================================
# DRUG HISTORY
# =====================================================

st.header("Medications Tried")

drug_list = df["drug_name"].dropna().unique().tolist()

num_drugs = st.number_input("Number of medications tried", 0, 10, 0)

drug_entries = []

for i in range(num_drugs):

    st.subheader(f"Medication {i+1}")

    drug_name = st.selectbox(
        f"Drug {i+1}",
        drug_list,
        key=f"drug_{i}"
    )

    rating = st.slider(
        f"Experience (1-5)",
        1, 5, 3,
        key=f"rating_{i}"
    )

    notes = st.text_area(
        f"Notes",
        key=f"notes_{i}"
    )

    drug_entries.append({
        "drug": drug_name,
        "rating": rating,
        "notes": notes
    })

# =====================================================
# RUN MODEL
# =====================================================

run = st.button("Analyze Experience")

if run:

    review_text = build_review_text(
    drug_entries)
    
    review_vec = embed(
    review_text)

    st.subheader("Patient Profile")

    ratings = [
    d["rating"]
    for d in drug_entries]
    
    if len(ratings) > 0:
        avg_rating = np.mean(ratings)
    
    else:
        avg_rating = 3

    has_review = int(
    review_text.strip() != "")

    # =================================================
    # PAST DRUGS
    # =================================================

    if len(drug_entries) > 0:

        st.subheader("PTBI for Previously Tried Medications")

        extracted_drugs = [d["drug"] for d in drug_entries]

        ptbi_df = compute_drug_ptbi(
            review_vec,
            extracted_drugs,
            diagnosis,
            gender,
            age,
            avg_rating,
            has_review)

        ratings_df = pd.DataFrame(drug_entries)[["drug", "rating"]]

        ptbi_df = ptbi_df.merge(ratings_df, on="drug", how="left")

        st.dataframe(ptbi_df)

    else:
        st.info("No medications entered.")

    # =================================================
    # RECOMMENDATIONS (FILTERED)
    # =================================================

    st.subheader("Recommended Medications")

    recs = recommend_drugs(
        review_vec,
        diagnosis,
        gender,
        age,
        avg_rating,
        has_review)

    st.dataframe(recs)

    best = recs.iloc[0]

    st.success(
        f"""


Best option:

**{best['drug']}**

PTBI: {best['predicted_PTBI']:.3f}

Uncertainty: ± {best['uncertainty']:.3f}

Range: [{best['lower_bound']:.3f}, {best['upper_bound']:.3f}]
"""
    )