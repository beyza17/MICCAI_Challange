# ==================== MAMA-MIA CHALLENGE FPIXEL SUBMISSION ====================
#Primary Tumour Segmentation (Task 1)


# ✅ 5 fold cross validation is implemented.

# Explanation is in "main.py" file.


import os
import subprocess
import numpy as np
import SimpleITK as sitk

class Model:
    def __init__(self, dataset, dataset_id="Dataset105_full_image", config="3d_fullres"):
        self.dataset = dataset  # <-- dataset object from Codabench
        self.dataset_id = dataset_id
        self.config = config
        self.predicted_segmentations = None

    def apply_breast_mask(self, seg_path, patient_id, output_path):
        segmentation = sitk.ReadImage(seg_path)
        segmentation_array = sitk.GetArrayFromImage(segmentation)

        # Read JSON metadata using the dataset object
        patient_info = self.dataset.read_json_file(patient_id)

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
        print("Running nnUNetv2 prediction...")

        nnUNet_raw = "/app/ingested_program/FPixel_submission"
        nnUNet_preprocessed = nnUNet_raw
        nnUNet_results = nnUNet_raw

        os.environ['nnUNet_raw'] = nnUNet_raw
        os.environ['nnUNet_preprocessed'] = nnUNet_preprocessed
        os.environ['nnUNet_results'] = nnUNet_results

        input_path = os.path.join(nnUNet_raw, self.dataset_id, "imagesTs")
        os.makedirs(output_dir, exist_ok=True)

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

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Prediction failed:\n", result.stderr)
            raise RuntimeError("nnUNetv2_predict failed")
        else:
            print("Prediction succeeded.\n", result.stdout)

        # Normalize filenames to lowercase
        for fname in os.listdir(output_dir):
            if fname.endswith(".nii.gz"):
                src = os.path.join(output_dir, fname)
                dst = os.path.join(output_dir, fname.lower())
                if src != dst:
                    os.rename(src, dst)

        # Post-processing: apply breast mask using dataset JSON
        print("Applying breast coordinate masking...")
        for fname in os.listdir(output_dir):
            if not fname.endswith(".nii.gz"):
                continue

            patient_id = fname.replace(".nii.gz", "")
            seg_path = os.path.join(output_dir, fname)
            final_seg_path = seg_path  # overwrite

            try:
                self.apply_breast_mask(seg_path, patient_id, final_seg_path)
            except Exception as e:
                print(f"WARNING: Failed to mask {patient_id}: {e}")

        self.predicted_segmentations = output_dir
        print(f"Final masked segmentations saved to: {output_dir}")
        return output_dir
