# ==================== MAMA-MIA CHALLENGE FPIXEL SUBMISSION ====================
#Primary Tumour Segmentation (Task 1)


#  5 fold cross validation is implemented.

# Explanation is in "main.py" file.
import os
# import sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)

import os
import subprocess
import numpy as np
import SimpleITK as sitk


class Model:
    def __init__(self, dataset, dataset_id="Dataset105_full_image", config="3d_fullres"):
        self.dataset = dataset
        self.dataset_id = dataset_id
        self.config = config
        self.predicted_segmentations = None

    def preprocess_images(self, preprocessed_output_dir):
        os.makedirs(preprocessed_output_dir, exist_ok=True)

        patient_ids = self.dataset.get_patient_id_list()
        for patient_id in patient_ids:
            print(f"Preprocessing {patient_id}...")

            phase_files = self.dataset.get_dce_mri_path_list(patient_id)

            for phase_idx, phase_file in enumerate(phase_files):
                image = sitk.ReadImage(phase_file, sitk.sitkFloat32)
                image = self.make_isotropic(image, interpolator=sitk.sitkBSpline)
                image = self.z_score_normalize(image)

                output_filename = f"{patient_id}_000{phase_idx}.nii.gz"
                sitk.WriteImage(image, os.path.join(preprocessed_output_dir, output_filename))

        print(" Preprocessing complete.")
        return preprocessed_output_dir

    @staticmethod
    def make_isotropic(image_sitk, target_spacing=1.0, interpolator=sitk.sitkBSpline):
        spacing = image_sitk.GetSpacing()
        size = image_sitk.GetSize()

        new_spacing = [target_spacing] * 3
        new_size = [int(round(size[i] * spacing[i] / new_spacing[i])) for i in range(3)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image_sitk.GetDirection())
        resampler.SetOutputOrigin(image_sitk.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetOutputPixelType(image_sitk.GetPixelID())
        resampler.SetInterpolator(interpolator)

        try:
            return resampler.Execute(image_sitk)
        except RuntimeError:
            identity_dir = (1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0)
            image_sitk.SetDirection(identity_dir)
            resampler.SetOutputDirection(identity_dir)
            return resampler.Execute(image_sitk)

    @staticmethod
    def z_score_normalize(image_sitk):
        array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
        mean = np.mean(array)
        std = np.std(array)
        norm_array = (array - mean) / std
        norm_image = sitk.GetImageFromArray(norm_array)
        norm_image.CopyInformation(image_sitk)
        return norm_image

    def apply_breast_mask(self, seg_path, patient_id, output_path):
        segmentation = sitk.ReadImage(seg_path)
        segmentation_array = sitk.GetArrayFromImage(segmentation)

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
        print(" Setting up directories...")

        # === Preprocessed input folder for nnUNet ===
        nnunet_input_images = os.path.join(output_dir, 'nnunet_input_images')
        os.makedirs(nnunet_input_images, exist_ok=True)

        # === Output folder for raw nnUNet segmentations ===
        output_dir_nnunet = os.path.join(output_dir, 'nnunet_seg')
        os.makedirs(output_dir_nnunet, exist_ok=True)

        # === Final output folder (masked segmentations) ===
        output_dir_final = os.path.join(output_dir, 'pred_segmentations')
        os.makedirs(output_dir_final, exist_ok=True)

        # === Step 1: Preprocess input images ===
        print(" Starting preprocessing...")
        self.preprocess_images(nnunet_input_images)

        # === Step 2: Run nnUNet prediction ===
        print(" Running nnUNetv2 prediction...")

        nnUNet_raw = "/app/ingested_program/sample_code_submission"
        os.environ['nnUNet_raw'] = nnUNet_raw
        os.environ['nnUNet_preprocessed'] = nnUNet_raw
        os.environ['nnUNet_results'] = nnUNet_raw

        cmd = [
            "nnUNetv2_predict",
            "-d", self.dataset_id,
            "-i", nnunet_input_images,
            "-o", output_dir_nnunet,
            "-f", "0", "1", "2", "3", "4",
            "-tr", "nnUNetTrainer",
            "-c", self.config,
            "-p", "nnUNetPlans"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(" Prediction failed:\n", result.stderr)
            raise RuntimeError("nnUNetv2_predict failed")
        else:
            print(" Prediction succeeded.\n", result.stdout)

        # === Step 3: Postprocess — apply breast mask ===
        print(" Applying breast coordinate masking...")

        for fname in os.listdir(output_dir_nnunet):
            if not fname.endswith(".nii.gz"):
                continue

            patient_id = fname.replace(".nii.gz", "")
            seg_path = os.path.join(output_dir_nnunet, fname)
            final_seg_path = os.path.join(output_dir_final, fname.lower())

            try:
                self.apply_breast_mask(seg_path, patient_id, final_seg_path)
            except Exception as e:
                print(f" WARNING: Failed to mask {patient_id}: {e}")

        self.predicted_segmentations = output_dir_final
        print(f" Final masked segmentations saved to: {output_dir_final}")
        return output_dir_final
