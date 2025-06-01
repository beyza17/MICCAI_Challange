from __init__ import Preprocessed
from model import Model

# Define paths
images_folder = "/app/ingested_program/FPixel_submission/original_images"
out_images_folder = "/app/ingested_program/FPixel_submission/Dataset006_miccai/imagesTs"

# Preprocess test images only
preprocessor = Preprocessed()
preprocessor.preprocess_all(images_folder, out_images_folder)

# preprocessed = Preprocessed()
# final_predictions_dir = preprocessed.isotropic("/app/ingested_program/FPixel_submission/imagesTs")# "labelsTs" is the prediction folder

model = Model(
        dataset_id="Dataset006_miccai",
        config="3d_fullres",
        patient_info_dir="/app/ingested_program/FPixel_submission/patient_info_files"
    )
final_predictions_dir = model.predict_segmentation("/app/ingested_program/FPixel_submission/labelsTs")# "labelsTs" is the prediction folder
print("Masked predictions are in:", final_predictions_dir) # postprocessed files are in "final_predictions_dir"
