from model import Model
from __init__ import Preprocessed
preprocessed = Preprocessed()
final_predictions_dir = preprocessed.isotropic("/app/ingested_program/FPixel_submission/imagesTs")# "labelsTs" is the prediction folder

model = Model(
        dataset_id="Dataset006_miccai",
        config="3d_fullres",
        patient_info_dir="/app/ingested_program/FPixel_submission/patient_info_files"
    )
final_predictions_dir = model.predict_segmentation("/app/ingested_program/FPixel_submission/labelsTs")# "labelsTs" is the prediction folder
print("Masked predictions are in:", final_predictions_dir) # postprocessed files are in "final_predictions_dir"
