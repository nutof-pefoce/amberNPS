import streamlit as st
import pandas as pd
import numpy as np
from mordred import Calculator, AdjacencyMatrix, Autocorrelation, EState, Weight
from rdkit import Chem
from rdkit.Chem import Draw
from mhfp.encoder import MHFPEncoder
import deepchem as dc
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Cache heavy objects
# ---------------------------
@st.cache_resource
def load_models():
    # Load pre-trained models
    cf_model = dc.models.MultitaskClassifier(n_tasks=1, n_classes=13, n_features=2048, layer_sizes=[256])
    cf_model.restore(checkpoint="cf_model_dir/checkpoint1.pt")

    rg_model = dc.models.MultitaskRegressor(n_tasks=3, n_features=6, layer_sizes=[20,10])
    rg_model.restore(checkpoint="rg_model_dir/checkpoint1.pt")
    
    return cf_model, rg_model

@st.cache_resource
def load_encoder_and_data():
    class_dataset = pd.read_csv('nps_classification.csv')
    le = LabelEncoder()
    class_dataset['labels'] = le.fit_transform(class_dataset['DrugClass'])
    return le, class_dataset

@st.cache_resource
def get_mhfp_encoder():
    return MHFPEncoder(n_permutations=4096, seed=42)

@st.cache_resource
def get_mordred_calc():
    calc = Calculator()
    calc.register(AdjacencyMatrix.AdjacencyMatrix('VR1'))
    calc.register(Autocorrelation.AATS(8, 'Z'))
    calc.register(Autocorrelation.AATS(3, 'i'))
    calc.register(Autocorrelation.ATSC(6, 's'))
    calc.register(Autocorrelation.ATSC(2, 'i'))
    calc.register(EState.AtomTypeEState('count', 'dsN'))
    return calc

def MordredCalculator(smiles, calc):
    mol = Chem.MolFromSmiles(smiles)
    return calc.pandas([mol])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title(':red[amber]NPS 🩸')
st.subheader('A QSAR-based app for the prediction of lethal blood concentration of New Psychoactive Substances', divider='red')

with st.form('SMILES_input_form'):
    smi = st.text_input('Enter SMILES', placeholder='example: CC(CC1=CC=CC=C1)N')
    st.caption("We recommend using Canonical SMILES available at [PubChem](https://pubchem.ncbi.nlm.nih.gov/)")
    go = st.form_submit_button("Calculate")

if smi and go:
    # Load resources
    cf_model, rg_model = load_models()
    le, class_dataset = load_encoder_and_data()
    mhfp_encoder = get_mhfp_encoder()
    calc = get_mordred_calc()

    # ---------------- Prediction: Classification ----------------
    fingerprints = np.array([mhfp_encoder.secfp_from_smiles(smi)], dtype=np.float32)
    prediction = cf_model.predict_on_batch(fingerprints)
    predicted_class = np.argmax(prediction, axis=2).flatten()
    predicted_class_name = le.inverse_transform(predicted_class)[0]

    # ---------------- Prediction: Regression ----------------
    features = MordredCalculator(smi, calc).values
    pred_reg = rg_model.predict_on_batch(features)

    molweight = Weight.Weight(True, False)(Chem.MolFromSmiles(smi))

    def convert_pLBC_to_LBC(pLBC):
        LBCmol = 10 ** -pLBC
        return LBCmol * molweight

    LOLBC = convert_pLBC_to_LBC(pred_reg[0][0][0])
    HOLBC = convert_pLBC_to_LBC(pred_reg[0][2][0])
    LBC50 = convert_pLBC_to_LBC(pred_reg[0][1][0])

    # ---------------- Results ----------------
    st.info(f"Assigned classification: {predicted_class_name}")
    if LBC50 > 1000:
        st.success(f"Predicted lethal blood concentration range: {LOLBC / 1000:.2f} to {HOLBC / 1000:.2f} μg/mL (median = {LBC50:.2f} μg/mL)")
    else:
        st.success(f"Predicted lethal blood concentration range: {LOLBC:.2f} to {HOLBC:.2f} ng/mL (median = {LBC50:.2f} ng/mL)")

    st.image(Draw.MolToImage(Chem.MolFromSmiles(smi)), caption='Molecular structure')

st.caption('Proudly developed in Ceará 🌵, Brazil 🇧🇷')

