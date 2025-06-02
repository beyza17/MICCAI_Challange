# ==================== MAMA-MIA CHALLENGE FPIXEL SUBMISSION ====================
#Primary Tumour Segmentation (Task 1)


# ✅ 5 fold cross validation is implemented, the prediction of each test data will take 1 minute per fold.

# Example usage: 
# model = Model(
#     dataset_id="Dataset006_miccai",
#     config="3d_fullres",
#     patient_info_dir="/app/ingested_program/FPixel_submission"
# )
# output_path = model.predict_segmentation("/path/to/save_predictions")


import os
import subprocess
import json
import shutil
import numpy as np
import SimpleITK as sitk

class Model:
    def __init__(self, dataset_id="Dataset006_miccai", config="3d_fullres", patient_info_dir=None):
        self.dataset_id = dataset_id
        self.config = config
        self.predicted_segmentations = None
        self.patient_info_dir = patient_info_dir  # path to JSON files like duke_001.json

    def apply_breast_mask(self, seg_path, json_path, output_path):
        segmentation = sitk.ReadImage(seg_path)
        segmentation_array = sitk.GetArrayFromImage(segmentation)

        with open(json_path, 'r') as f:
            patient_info = json.load(f)

        coords = patient_info["primary_lesion"]["breast_coordinates"]
        x_min, x_max = coords["x_min"], coords["x_max"]
        y_min, y_max = coords["y_min"], coords["y_max"]
        z_min, z_max = coords["z_min"], coords["z_max"]

        masked_segmentation = np.zeros_like(segmentation_array)
        masked_segmentation[x_min:x_max, y_min:y_max, z_min:z_max] = \
            segmentation_array[x_min:x_max, y_min:y_max, z_min:z_max]

        masked_seg_image = sitk.GetImageFromArray(masked_segmentation)
        masked_seg_image.CopyInformation(segmentation)
        sitk.WriteImage(masked_seg_image, output_path)

    def predict_segmentation(self, output_dir):
        # === Set nnUNet environment variables ===
        nnUNet_raw = "/app/ingested_program/FPixel_submission"
        nnUNet_preprocessed ="/app/ingested_program/FPixel_submission"
        nnUNet_results ="/app/ingested_program/FPixel_submission"

        os.environ['nnUNet_raw'] = nnUNet_raw
        os.environ['nnUNet_preprocessed'] = nnUNet_preprocessed
        os.environ['nnUNet_results'] = nnUNet_results

        # === Prepare input and output paths ===
        input_path = os.path.join(nnUNet_raw, self.dataset_id, "imagesTs")
        os.makedirs(output_dir, exist_ok=True)

        # === Run nnUNetv2 prediction ===
        cmd = [
            "nnUNetv2_predict",
            "-d", self.dataset_id,
            "-i", input_path,
            "-o", output_dir,
            "-f", "0", "1", "2", "3", "4",
            "-tr", "nnUNetTrainer",
            "-c", self.config,
            "-p", "nnUNetPlans"
        ]

        print("Running nnUNetv2 prediction...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Prediction failed:\n", result.stderr)
            raise RuntimeError("nnUNetv2_predict failed")
        else:
            print("Prediction succeeded.\n", result.stdout)

        # === Normalize filenames to lowercase (due to json file names) ===
        for fname in os.listdir(output_dir):
            if fname.endswith(".nii.gz"):
                src = os.path.join(output_dir, fname)
                dst = os.path.join(output_dir, fname.lower())
                if src != dst:
                    os.rename(src, dst)

        # === Postprocessing: apply breast mask using JSON ===
        print("Applying breast coordinate masking...")
        output_masked_dir = os.path.join(output_dir, "pred_segmentations") 
        os.makedirs(output_masked_dir, exist_ok=True)

        for fname in os.listdir(output_dir):
            if not fname.endswith(".nii.gz"):
                continue

            patient_id = fname.replace(".nii.gz", "")
            seg_path = os.path.join(output_dir, fname)
            json_path = os.path.join(self.patient_info_dir, f"{patient_id}.json")
            final_seg_path = os.path.join(output_masked_dir, f"{patient_id}.nii.gz")

            if not os.path.exists(json_path):
                print(f"WARNING: Missing JSON for {patient_id}, skipping mask.")
                shutil.copy(seg_path, final_seg_path)  # still save unmasked version
                continue

            self.apply_breast_mask(seg_path, json_path, final_seg_path)

        self.predicted_segmentations = output_masked_dir
        print(f"Final masked segmentations saved to: {output_masked_dir}")
        return output_masked_dir










