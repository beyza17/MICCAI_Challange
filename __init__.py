


import os
from preprocessing import Preprocessed
from model import Model

class SubmissionModel:
    def __init__(self,dataset):
        # Define paths
        self.dataset_id = "Dataset105_full_image"  # <-- match Codabench ingested name
        self.preprocessed_folder = f"/app/ingested_program/{self.dataset_id}/imagesTs"
        self.prediction_output_dir = "/app/predictions/pred_segmentations"

        # Ensure required folders exist
        os.makedirs(self.preprocessed_folder, exist_ok=True)
        os.makedirs(self.prediction_output_dir, exist_ok=True)

        # Preprocess test images
        print("Starting preprocessing of test images...")
        preprocessor = Preprocessed()
        preprocessor.preprocess_all(self.preprocessed_folder)

        # Create model instance
        self.model = Model(
            dataset=dataset,
            dataset_id="Dataset105_full_image",
            config="3d_fullres")

    def predict(self):
        print("Starting prediction...")
        self.model.predict_segmentation(self.prediction_output_dir)
