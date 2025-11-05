import math
import numpy as np
import pandas as pd
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Draw

from mordred import Calculator, descriptors, AdjacencyMatrix, Autocorrelation, EState, DistanceMatrix, TopologicalIndex, BCUT, MoeType, RingCount, BaryszMatrix, ExtendedTopochemicalAtom, TopologicalCharge, Weight

import streamlit as st

@st.cache_resource
with open('multitask_regressor.pkl', 'rb') as f:
    mlp = pickle.load(f)

@st.cache_resource
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
@st.cache_resource
with open('random_forest_model.pkl', 'rb') as f:
    rf = pickle.load(f)

@st.cache_resource
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@st.cache_resource
def MACCS_Generator(smi):
  mol = Chem.MolFromSmiles(smi)
  maccs = MACCSkeys.GenMACCSKeys(mol)
  maccs_list = list(maccs)
  maccs_name = [f'MACCS_{i}' for i in range(1, 167)]
  maccs_df = pd.DataFrame([maccs_list[1:]], columns=maccs_name)
  return maccs_df

@st.cache_resource
def MordredCalculator(smi):
  mol = Chem.MolFromSmiles(smi)
  calc = Calculator()
  # Register VR1_A
  calc.register(AdjacencyMatrix.AdjacencyMatrix('VR1'))
  # Register ATSC0p
  calc.register(Autocorrelation.ATSC(0, 'p'))
  # Register ATSC2i
  calc.register(Autocorrelation.ATSC(2, 'i'))
  # Register SpMax_D
  calc.register(DistanceMatrix.DistanceMatrix('SpMax'))
  # Register NdCH2
  calc.register(EState.AtomTypeEState('count', 'dCH2'))
  # Register SaaaC
  calc.register(EState.AtomTypeEState('sum', 'aaaC'))
  # Register EState_VSA2
  calc.register(MoeType.EState_VSA(2))
  # Register n11FARing
  calc.register(RingCount.RingCount(11, False, True, True, None))
  # Register ATS5s
  calc.register(Autocorrelation.ATS (5, 's'))
  # Register ATSC0Z
  calc.register(Autocorrelation.ATSC (0, 'Z'))
  # Register BCUTd-1l
  calc.register(BCUT.BCUT('d', -1))
  # Register SpDiam_Dzare
  calc.register(BaryszMatrix.BaryszMatrix('are', 'SpDiam'))
  # Register SdCH2
  calc.register(EState.AtomTypeEState ('sum', 'dCH2'))
  # Register AETA_beta_ns_d
  calc.register(ExtendedTopochemicalAtom.EtaVEMCount('ns_d', True))
  # Register n11FAHRing
  calc.register(RingCount.RingCount(11, False, True, False, True))
  # Register GGI9
  calc.register(TopologicalCharge.TopologicalCharge('raw',9))
  # Register VR2_A
  calc.register(AdjacencyMatrix.AdjacencyMatrix('VR2'))
  # Register ATSC5s
  calc.register(Autocorrelation.ATSC(5, 's'))
  # Register AATSC1s
  calc.register(Autocorrelation.AATSC(1, 's'))
  # Register GATS7s
  calc.register(Autocorrelation.GATS(7, 's'))
  # Register SpMAD_Dzare
  calc.register(BaryszMatrix.BaryszMatrix('are', 'SpMAD'))
  # Register SdO
  calc.register(EState.AtomTypeEState('sum', 'dO'))
  # Register PEOE_VSA13
  calc.register(MoeType.PEOE_VSA(13))
  # Register VSA_EState8
  calc.register(MoeType.VSA_EState(8))
  # Register n5aRing
  calc.register(RingCount.RingCount(5, True, False, False, None))
  # Register Diameter and PetitjeanIndex
  calc.register(TopologicalIndex.Diameter())
  calc.register(TopologicalIndex.PetitjeanIndex())
  # Register MolWeight
  calc.register(Weight.Weight(True,False))
  result = np.array(calc(mol))
  features = result[:-1]
  mw = result[-1]
  return features, mw

@st.cache_resource      
def convert_pLBC_to_LBC(pLBC, mw):
    LBCmol = 10 ** -pLBC
    LBC = LBCmol * mw
    return LBC

st.title(':red[amber]NPS 🩸')
st.subheader('A QSAR-based app for the prediction of lethal blood concentration of New Psychoactive Substances', divider='red')

with st.form('SMILES_input_form'):

    smi = st.text_input('Enter SMILES',placeholder='example: CC(CC1=CC=CC=C1)N')
    st.caption("We recommend using Canonical SMILES available at [PubChem](https://pubchem.ncbi.nlm.nih.gov/)")
    
    col1, col2 = st.columns([9, 13])
    
    with col2:
        go = st.form_submit_button("Calculate")        
    
mol = Chem.MolFromSmiles(smi)
img = Draw.MolToImage(mol)
if smi or go:
    
    ## MORDRED CALCs
    # Create empty Calculator instance
    features, mw = MordredCalculator(smi)
    maccs_keys = MACCS_Generator(smi)

    # Wait for the Weka process to finish
    with st.spinner('Operation in progress'):
        drug_class = rf.predict(maccs_keys)
        pred = mlp.predict(features)
    
    # Retrieve the result from the queue
    pLOLBC = pred[0, 0]
    pLBC50 = pred[0, 1]
    pHOLBC = pred[0, 2]

    # Continue processing the result or displaying it in the Streamlit app
    drug_class = le.inverse_transform(drug_class)[0]
    
    LBC50 = convert_pLBC_to_LBC(pLBC50, mw)
    LOLBC = convert_pLBC_to_LBC(pLOLBC, mw)
    HOLBC = convert_pLBC_to_LBC(pHOLBC, mw)

    
    st.info(f"Assigned classification: {drug_class}")
    if LBC50 > 1000:
        st.success(f"Predicted lethal blood concentration range: {LOLBC / 1000:.2f} to {HOLBC / 1000:.2f} μg/mL (median = {LBC50:.2f} μg/mL)")
    else:
        st.success(f"Predicted lethal blood concentration range: {LOLBC:.2f} to {HOLBC:.2f} ng/mL (median = {LBC50:.2f} ng/mL)")
    
    col4, col5 = st.columns([5, 12])        
    with col5:
        st.image(img, caption='Molecular structure')
    
col6, col7 = st.columns([5, 11])
with col7:
    st.caption('Please cite: [Correa et al. - Emerging Trends in Drugs, Addictions, and Health - 2024](https://doi.org/10.1016/j.etdah.2024.100156)')
