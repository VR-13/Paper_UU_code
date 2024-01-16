# Thesis_UU

The data folder contains the raw data for the patient, therapist and observer. The features are in the folders 'patient, therapist and observer' and the corresponding WAI scores are in a different folder.
The features all contain a ppnr and session with correspond with the WAI scores. 

Features are both in individual files and in a combined file. While the models run on the combined files, the individual feature csv files are included as well so they can be updated. 
For instance:
- whisper needs to be run again with v2-large model
- audio and text models should be finetuned on dutch similar dataset to ours

the code used for the models, including hyperparameter optimilization, is included in the 'model code' folder as notebooks for easy use on colab.
the code for the features can be found in the 'features' folder.
