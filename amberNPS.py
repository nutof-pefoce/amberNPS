import os
import streamlit as st

import math
import multiprocessing

import os

import random
import pandas as pd
import numpy as np
import requests
from mordred import Calculator, descriptors, AdjacencyMatrix, Autocorrelation, EState, Weight
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from mhfp.encoder import MHFPEncoder
import deepchem as dc
from deepchem.models import MultitaskClassifier, MultitaskRegressor
from deepchem.models.optimizers import Adam
from deepchem.trans import BalancingTransformer
from sklearn.preprocessing import LabelEncoder

      
if __name__ == "__main__":
    
    st.title(':red[amber]NPS 🩸')
    st.subheader('A QSAR-based app for the prediction of lethal blood concentration of New Psychoactive Substances', divider='red')
    
    with st.form('SMILES_input_form'):
    
        smi = st.text_input('Enter SMILES',placeholder='example: CC(CC1=CC=CC=C1)N')
        st.caption("We recommend using Canonical SMILES available at [PubChem](https://pubchem.ncbi.nlm.nih.gov/)")
        
        col1, col2 = st.columns([9, 13])
        
        with col2:
            go = st.form_submit_button("Calculate")        

 
    class_dataset = pd.read_csv('nps_classification.csv')
    amber_dataset = pd.read_csv('amber_nps.csv')
    reg_dataset = pd.read_csv('TSS3.csv')

    # Define the SMILES column and target columns
    smiles_column_cf = class_dataset['SMILES']


    le = LabelEncoder()
    class_dataset['labels'] = le.fit_transform(class_dataset['DrugClass'])
    labels = class_dataset['labels'].values

    # Define the featurizer
    mhfp_encoder = MHFPEncoder(n_permutations=4096, seed = 42)

    # 1. Load CSV data using Pandas

    X_fingerprints = np.array([mhfp_encoder.secfp_from_smiles(smi) for smi in smiles_column_cf], dtype=np.float32)

    # Create a DeepPurpose dataset
    cf_dataset = dc.data.NumpyDataset(X_fingerprints, y=labels)
    balancer = BalancingTransformer(cf_dataset)
    cf_dataset = balancer.transform(cf_dataset)

    cf_model = MultitaskClassifier(n_tasks=1, n_classes=13, n_features=2048, layer_sizes=[256], activation_fns='relu', dropouts=0.02, weight_decay_penalty=0.02)

    cf_model.fit(cf_dataset, nb_epoch=75)


    calc = Calculator()

    # Register VR1_A
    calc.register(AdjacencyMatrix.AdjacencyMatrix('VR1'))
    # Register AATS8Z
    calc.register(Autocorrelation.AATS (8, 'Z'))
    # Register AATS3i
    calc.register(Autocorrelation.AATS (3, 'i'))
    # Register ATSC6s
    calc.register(Autocorrelation.ATSC (6, 's'))
    # Register ATSC2i
    calc.register(Autocorrelation.ATSC (2, 'i'))
    # Register NdsN
    calc.register(EState.AtomTypeEState ('count', 'dsN'))
          
  
    def MordredCalculator(smiles_column):
        molecule = smiles_column.apply(Chem.MolFromSmiles)
        result = calc.pandas(molecule)
        return result

    smiles_column = reg_dataset['SMILES']
    targets = reg_dataset[['pLOLBC', 'pLBC50', 'pHOLBC']].values


    #Descriptor calculation
    descriptors = MordredCalculator(smiles_column)

    #X_features = pd.concat([descriptors, encoded_df], axis=1)
    X_features = descriptors
    X_features = X_features.values

    rg_dataset = dc.data.NumpyDataset(X_features, y=targets, ids=smiles_column)

    rg_model = MultitaskRegressor(n_tasks=3, n_features=6, layer_sizes=[20,10], activation_fns='softplus', weight_decay_penalty=0.1)

    rg_model.fit(rg_dataset, nb_epoch=600)



    mol = Chem.MolFromSmiles(smi)
    img = Draw.MolToImage(mol)



    if smi or go:
        
        fingerprints = np.array([mhfp_encoder.secfp_from_smiles(smi)], dtype=np.float32)

        # Make prediction using the trained model
        prediction = cf_model.predict_on_batch(fingerprints)

        # Get predicted class label (assuming it's a classification model)
        predicted_class = np.argmax(prediction, axis=2).flatten()

        # Decode the predicted class label if necessary (using le.inverse_transform)
        predicted_class_name = le.inverse_transform([predicted_class][0])[0]

        smiles_series = pd.Series(smi)

        # Featurize the molecule
        features = MordredCalculator(smiles_series)

        # Make prediction using the trained model
        prediction = model.predict_on_batch(features)

        molweight = Weight.Weight(True,False)
        mw = molweight(Chem.MolFromSmiles(smi))

        def convert_pLBC_to_LBC(pLBC):
                    LBCmol = 10 ** -pLBC
                    LBC = LBCmol * mw
                    return LBC

        LOLBC = convert_pLBC_to_LBC(prediction[0][0][0])
        HOLBC = convert_pLBC_to_LBC(prediction[0][2][0])
        LBC50 = convert_pLBC_to_LBC(prediction[0][1][0])

        # Wait for the Weka process to finish
        #with st.spinner('Operation in progress'):
            #weka_proc.join()
        
            
               
        st.info(f"Assigned classification: {predicted_class_name}")
        
        if LBC > 1000:
            st.success(f"\n Predicted lethal blood concentration range: {LOLBC / 1000:.2f} to {HOLBC / 1000:.2f} μg/mL (median = {LBC50:.2f} μg/mL)")
        else:
            st.success(f"\nPredicted lethal blood concentration range: {LOLBC:.2f} to {HOLBC:.2f} ng/mL (median = {LBC50:.2f} ng/mL)")
        
        col4, col5 = st.columns([5, 12])        
        with col5:
            st.image(img, caption='Molecular structure')
        
    col6, col7 = st.columns([5, 11])
    with col7:
<<<<<<< HEAD
        st.caption('Proudly developed in Ceará 🌵, Brazil 🇧🇷')

=======
        st.caption('Proudly developed in Ceará 🌵, Brazil 🇧🇷')
>>>>>>> 9fb59ae (Local changes and new files)
