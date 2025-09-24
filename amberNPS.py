import os
import streamlit as st

@st.cache_resource # or st.cache_data for older Streamlit versions
def install_java():
        # Example for downloading and extracting Java 11
        # Adjust URL and extraction command for your specific OpenJDK version
    os.system("wget -O /tmp/jdk.tar.gz https://download.java.net/java/GA/jdk11/97a06277a0664805b54194e43e74257c/12/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz")
    os.system("mkdir -p /app/jdk")
    os.system("tar -xzf /tmp/jdk.tar.gz -C /app/jdk --strip-components=1")
    os.environ["JAVA_HOME"] = "/app/jdk"
    os.environ["PATH"] = f"{os.environ['JAVA_HOME']}/bin:{os.environ['PATH']}"
    st.success("Java installed and environment variables set.")
   
install_java()


import math
import multiprocessing

from rdkit import Chem
from rdkit.Chem import Draw

from mordred import Calculator, descriptors, AdjacencyMatrix, Autocorrelation, EState, AcidBase, InformationContent, RotatableBond, Weight



def weka_process(results_queue):
    import weka.core.jvm as jvm
    jvm.start(packages=True, auto_install=True)
    import weka.core.packages as packages
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
    
    ## pBLC prediction task

    result_list = list(result2.values())
    result_list.append(missing_value())

    inst2 = Instance.create_instance(result_list, weight=1.0)
    inst2.dataset = data

    data_file = "QSAR_NPS_training_set_v.2.csv"
    model_file = "MLPReg_QSAR_NPS_v.2.model"

    # load
    data = load_any_file(data_file)
    data.class_is_last()

    ## train classifier
    cls = Classifier(classname="weka.classifiers.functions.MLPRegressor")
    #cls.build_classifier(data)

    ## save model
    #cls.serialize(model_file, header=data)

    # load model
    model, header = Classifier.deserialize(model_file)

    values = result_list 
    inst = Instance.create_instance(values, weight=1.0)
    inst.dataset = header

    # make prediction for new instance
    pLBC = model.classify_instance(inst)

    jvm.stop()
    
    # Obtain results from Weka-related code...
    weka_result = clsf, pLBC 

    # Put the result in the queue
    results_queue.put(weka_result) 
      
if __name__ == "__main__":
    
    st.title(':red[amber]NPS :drop_of_blood:')
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
        calc1 = Calculator()
        calc2 = Calculator()

        # GATS1s
        calc1.register(Autocorrelation.GATS(1,'s'))
        # nBase
        calc1.register(AcidBase.BasicGroupCount())
        # SsssN
        calc1.register(EState.AtomTypeEState('sum','sssN'))
        #SsOH
        calc1.register(EState.AtomTypeEState('sum','sOH'))
        #AATS5p
        calc1.register(Autocorrelation.AATS (5, 'p'))
        #GATS4s
        calc1.register(Autocorrelation.GATS (4, 's'))
        #AATSC3s 
        calc1.register(Autocorrelation.AATSC (3, 's'))
        #IC5
        calc1.register(InformationContent.InformationContent (5))
        #NsOH 
        calc1.register(EState.AtomTypeEState('count','sOH'))
        #nRot 
        calc1.register(RotatableBond.RotatableBondsCount())

        result1 = calc1(mol)

        # Register VR1_A
        calc2.register(AdjacencyMatrix.AdjacencyMatrix('VR1'))
        # Register AATS8Z
        calc2.register(Autocorrelation.AATS (8, 'Z'))
        # Register AATS3i
        calc2.register(Autocorrelation.AATS (3, 'i'))
        # Register ATSC6s
        calc2.register(Autocorrelation.ATSC (6, 's'))
        # Register ATSC2i
        calc2.register(Autocorrelation.ATSC (2, 'i'))
        # Register NdsN
        calc2.register(EState.AtomTypeEState ('count', 'dsN'))
                    
        result2 = calc2(mol)
        
        molweight = Weight.Weight(True,False)
        mw = molweight(mol)
        
        ## END of MORDRED CALCs
        
        # Create a Queue for inter-process communication
        results_queue = multiprocessing.Queue()
        
        # Start the Weka-related process
        weka_proc = multiprocessing.Process(target=weka_process, args=(results_queue,))
        
        weka_proc.start()   

        # Wait for the Weka process to finish
        with st.spinner('Operation in progress'):
            weka_proc.join()
        
        
        # Retrieve the result from the queue
        weka_result = results_queue.get()

        # Continue processing the result or displaying it in the Streamlit app
        clsf, pLBC = weka_result
        
        
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

        # Convert pH to [H+]
        LBC = convert_pLBC_to_LBC(pLBC)
                
        ## st.write(f"[LBC] for pLBC {pLBC}: {round(LBC, 4)}")

        # Calculate range around pH +/- 1
        lower_range, upper_range = calculate_range_around_pLBC(pLBC)
        
        st.info(f"Assigned classification: {clsf}")
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
