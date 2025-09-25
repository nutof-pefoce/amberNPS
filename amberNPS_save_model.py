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
cf_model.save_checkpoint(model_dir="cf_model_dir")


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
rg_model.save_checkpoint(model_dir="rg_model_dir")