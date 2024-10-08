import math
import random
import concurrent.futures
import atexit
import threading

from rdkit import Chem
from rdkit.Chem import Draw

from mordred import Calculator, descriptors, AdjacencyMatrix, Autocorrelation, EState, AcidBase, InformationContent, RotatableBond, Weight

import streamlit as st

# Cache the JVM start function to ensure it's only initialized once
@st.cache_resource
def start_jvm():
    import weka.core.jvm as jvm
    if not jvm.started:
        jvm.start(packages=True, auto_install=True)
    return jvm

# Function to stop the JVM, ensuring it is done on the main thread
def stop_jvm():
    import weka.core.jvm as jvm
    if jvm.started and threading.current_thread() is threading.main_thread():
        jvm.stop()

# Register the JVM stop function to run when the script exits
atexit.register(stop_jvm)

def weka_process(result1, result2):
    from weka.core.converters import load_any_file
    from weka.classifiers import Classifier
    from weka.core.dataset import Instance, missing_value

    ## Class assignment task
    data_file = "NPS_Class.arff"
    data = load_any_file(data_file)
    data.class_is_last()

    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(data)

    header = Classifier

    values = list(result1.values())
    values.append(missing_value())

    inst1 = Instance.create_instance(values, weight=1.0)
    inst1.dataset = data

    pred = cls.classify_instance(inst1)
    clsf = inst1.class_attribute.value(int(pred))

    ## pLBC prediction task
    result_list = list(result2.values())
    result_list.append(missing_value())

    inst2 = Instance.create_instance(result_list, weight=1.0)
    inst2.dataset = data

    data_file = "QSAR_NPS_training_set_v.2.csv"
    model_file = "MLPReg_QSAR_NPS_v.2.model"

    # load
    data = load_any_file(data_file)
    data.class_is_last()

    ## Load model
    cls = Classifier(classname="weka.classifiers.functions.MLPRegressor")
    model, header = cls.deserialize(model_file)

    values = result_list
    inst = Instance.create_instance(values, weight=1.0)
    inst.dataset = header

    # make prediction for new instance
    pLBC = model.classify_instance(inst)

    return clsf, pLBC

# Start the JVM at the beginning
start_jvm()

if __name__ == "__main__":
    st.title(':red[amber]NPS :drop_of_blood:')
    st.subheader('A QSAR-based app for the prediction of lethal blood concentration of New Psychoactive Substances', divider='red')

    with st.form('SMILES_input_form'):
        smi = st.text_input('Enter SMILES')
        st.caption("We recommend using Canonical SMILES available at [PubChem](https://pubchem.ncbi.nlm.nih.gov/)")
        col1, col2 = st.columns([9, 13])
        with col2:
            go = st.form_submit_button("Calculate")        

    if go:
        mol = Chem.MolFromSmiles(smi)
        img = Draw.MolToImage(mol)
        ## MORDRED CALCs
        calc1 = Calculator()
        calc2 = Calculator()

        # GATS1s
        calc1.register(Autocorrelation.GATS(1, 's'))
        # nBase
        calc1.register(AcidBase.BasicGroupCount())
        # SsssN
        calc1.register(EState.AtomTypeEState('sum', 'sssN'))
        # SsOH
        calc1.register(EState.AtomTypeEState('sum', 'sOH'))
        # AATS5p
        calc1.register(Autocorrelation.AATS(5, 'p'))
        # GATS4s
        calc1.register(Autocorrelation.GATS(4, 's'))
        # AATSC3s 
        calc1.register(Autocorrelation.AATSC(3, 's'))
        # IC5
        calc1.register(InformationContent.InformationContent(5))
        # NsOH 
        calc1.register(EState.AtomTypeEState('count', 'sOH'))
        # nRot 
        calc1.register(RotatableBond.RotatableBondsCount())

        result1 = calc1(mol)

        # Register VR1_A
        calc2.register(AdjacencyMatrix.AdjacencyMatrix('VR1'))
        # Register AATS8Z
        calc2.register(Autocorrelation.AATS(8, 'Z'))
        # Register AATS3i
        calc2.register(Autocorrelation.AATS(3, 'i'))
        # Register ATSC6s
        calc2.register(Autocorrelation.ATSC(6, 's'))
        # Register ATSC2i
        calc2.register(Autocorrelation.ATSC(2, 'i'))
        # Register NdsN
        calc2.register(EState.AtomTypeEState('count', 'dsN'))

        result2 = calc2(mol)

        molweight = Weight.Weight(True, False)
        mw = molweight(mol)

        # Run the Weka-related function directly
        with st.spinner('Operation in progress'):
            clsf, pLBC = weka_process(result1, result2)

        def convert_pLBC_to_LBC(pLBC):
            LBCmol = 10 ** -pLBC
            LBC = LBCmol * mw
            return LBC

        def calculate_range_around_pLBC(pLBC):
            if clsf == "Cannabinoids" or clsf == "Benzodiazepines": 
                diff = 0.57
            elif clsf == "Phenethylamines" or clsf == "Opioids" or clsf == "Cathinones":
                diff = 0.75
            else:
                diff = 0.75
            lower_pLBC = pLBC - diff
            upper_pLBC = pLBC + diff
            lower_LBC = convert_pLBC_to_LBC(upper_pLBC) 
            upper_LBC = convert_pLBC_to_LBC(lower_pLBC)
            return lower_LBC, upper_LBC

        # Convert pLBC to LBC
        LBC = convert_pLBC_to_LBC(pLBC)

        # Calculate range around pLBC
        lower_range, upper_range = calculate_range_around_pLBC(pLBC)

        st.info(f"Assigned classification: {clsf}")
        st.success(f"Predicted pLBC: {round(pLBC, 4)}")
        if LBC > 1000:
            st.success(f"Predicted lethal blood concentration range: {round(lower_range / 1000, 2)} to {round(upper_range / 1000, 2)} μg/mL")
        else:
            st.success(f"Predicted lethal blood concentration range: {round(lower_range, 2)} to {round(upper_range, 2)} ng/mL")

        col4, col5 = st.columns([5, 12])        
        with col5:
            st.image(img, caption='Molecular structure')

    col6, col7 = st.columns([5, 11])
    with col7:
        st.caption('Proudly developed in Ceará :cactus:, Brazil :flag-br:')
