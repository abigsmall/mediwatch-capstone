import json
import os

import requests
import streamlit as st

from mediwatch.settings import settings


API_URL = os.getenv("MEDIWATCH_API_URL", "http://api:8000")


DEFAULT_PAYLOAD = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 4,
    "num_lab_procedures": 41,
    "num_procedures": 0,
    "num_medications": 16,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": "428",
    "diag_2": "250.83",
    "diag_3": "401",
    "number_diagnoses": 9,
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "metformin": "No",
    "insulin": "Up",
    "change": "Ch",
    "diabetesMed": "Yes",
}


st.set_page_config(page_title="MediWatch", page_icon=":hospital:", layout="wide")
st.title("MediWatch")
st.caption("Predict patient readmission risk and demo the production API.")

payload_text = st.text_area(
    "Patient feature payload (JSON)",
    value=json.dumps(DEFAULT_PAYLOAD, indent=2),
    height=420,
)

if st.button("Predict Readmission Risk", type="primary"):
    try:
        payload = json.loads(payload_text)
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        st.success("Prediction complete.")
        st.metric("Readmission Probability", f"{result['probability']:.3f}")
        st.write(result)
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Prediction failed: {exc}")

st.subheader("Expected dataset location")
st.code(str(settings.raw_dataset_path))
