# ==================== MAMA-MIA CHALLENGE FPIXEL SUBMISSION ====================
# Primary Tumour Segmentation (Task 1)

import os
import numpy as np
import shutil
import SimpleITK as sitk
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import logging
from skimage.measure import label, regionprops

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def keep_largest_component(mask_array):
    labeled = label(mask_array)
    if labeled.max() == 0:
        return mask_array
    regions = regionprops(labeled)
    largest = max(regions, key=lambda x: x.area)
    return (labeled == largest.label).astype(np.uint8)

class Model:
    def __init__(self, dataset, dataset_id="Dataset105_full_image", config="3d_fullres"):
        self.dataset = dataset
        self.dataset_id = dataset_id
        self.config = config
        self.predicted_segmentations = None

        # Set nnUNet environment
        nnUNet_raw = "/app/ingested_program/sample_code_submission"
        os.environ['nnUNet_raw'] = nnUNet_raw
        os.environ['nnUNet_preprocessed'] = nnUNet_raw
        os.environ['nnUNet_results'] = nnUNet_raw

        logger.info(f"nnUNet environment set to: {nnUNet_raw}")

    def preprocess_images(self, preprocessed_output_dir):
        os.makedirs(preprocessed_output_dir, exist_ok=True)
        patient_ids = self.dataset.get_patient_id_list()

        logger.info(f"Found {len(patient_ids)} patients: {patient_ids}")

        successful_preprocessing = 0
        for patient_id in patient_ids:
            try:
                logger.info(f"Preprocessing {patient_id}...")
                phase_files = self.dataset.get_dce_mri_path_list(patient_id)
                logger.info(f"Phase files for {patient_id}: {len(phase_files)} files")

                if len(phase_files) < 3:
                    logger.warning(f"Skipping {patient_id}: only {len(phase_files)} phases available")
                    continue

                for i in range(3):  # Use the first 3 phases
                    image_path = phase_files[i]
                    logger.info(f"Reading phase {i}: {image_path}")

                    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
                    image = self.make_isotropic(image, interpolator=sitk.sitkBSpline)

                    norm_array = sitk.GetArrayFromImage(image)
                    logger.info(f"Phase {i} stats after resampling - Min: {norm_array.min():.3f}, Max: {norm_array.max():.3f}, Mean: {norm_array.mean():.3f}")

                    output_filename = f"{patient_id}_{i:04d}.nii.gz"
                    output_path = os.path.join(preprocessed_output_dir, output_filename)
                    sitk.WriteImage(image, output_path)
                    logger.info(f"Saved preprocessed image: {output_filename}")

                successful_preprocessing += 1

            except Exception as e:
                logger.error(f"Error preprocessing {patient_id}: {str(e)}")
                continue

        logger.info(f"Preprocessing complete. Successfully processed {successful_preprocessing}/{len(patient_ids)} patients.")
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
        except RuntimeError as e:
            logger.warning(f"Resampling failed with direction matrix, trying with identity: {e}")
            identity_dir = (1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0)
            image_sitk.SetDirection(identity_dir)
            resampler.SetOutputDirection(identity_dir)
            return resampler.Execute(image_sitk)

    def resample_to_original_space(self, prediction_path, patient_id, output_path):
        try:
            prediction = sitk.ReadImage(prediction_path)
            phase_files = self.dataset.get_dce_mri_path_list(patient_id)
            original_image = sitk.ReadImage(phase_files[1])

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(original_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)

            resampled_prediction = resampler.Execute(prediction)
            sitk.WriteImage(resampled_prediction, output_path)
            return output_path

        except Exception as e:
            logger.error(f"Error resampling prediction for {patient_id}: {str(e)}")
            original_image = sitk.ReadImage(phase_files[1])
            empty_seg = sitk.Image(original_image.GetSize(), sitk.sitkUInt8)
            empty_seg.CopyInformation(original_image)
            sitk.WriteImage(empty_seg, output_path)
            return output_path

    def apply_breast_mask(self, seg_path, patient_id, output_path):
        try:
            segmentation = sitk.ReadImage(seg_path)
            segmentation_array = sitk.GetArrayFromImage(segmentation)

            patient_info = self.dataset.read_json_file(patient_id)
            coords = patient_info["primary_lesion"]["breast_coordinates"]
            x_min, x_max = coords["x_min"], coords["x_max"]
            y_min, y_max = coords["y_min"], coords["y_max"]
            z_min, z_max = coords["z_min"], coords["z_max"]

            masked_segmentation = np.zeros_like(segmentation_array)
            z_max = min(z_max, segmentation_array.shape[0])
            y_max = min(y_max, segmentation_array.shape[1])
            x_max = min(x_max, segmentation_array.shape[2])
            z_min = max(z_min, 0)
            y_min = max(y_min, 0)
            x_min = max(x_min, 0)

            masked_segmentation[z_min:z_max, y_min:y_max, x_min:x_max] = \
                segmentation_array[z_min:z_max, y_min:y_max, x_min:x_max]

            masked_seg_image = sitk.GetImageFromArray(masked_segmentation)
            masked_seg_image.CopyInformation(segmentation)
            sitk.WriteImage(masked_seg_image, output_path)

        except Exception as e:
            logger.error(f"Error applying breast mask for {patient_id}: {str(e)}")
            segmentation = sitk.ReadImage(seg_path)
            sitk.WriteImage(segmentation, output_path)

    def predict_segmentation(self, output_dir):
        logger.info("Setting up directories...")

        nnunet_input_images = os.path.join(output_dir, 'nnunet_input_images')
        os.makedirs(nnunet_input_images, exist_ok=True)

        output_dir_nnunet = os.path.join(output_dir, 'nnunet_seg')
        os.makedirs(output_dir_nnunet, exist_ok=True)

        output_dir_final = os.path.join(output_dir, 'pred_segmentations')
        os.makedirs(output_dir_final, exist_ok=True)

        self.preprocess_images(nnunet_input_images)

        preprocessed_files = [f for f in os.listdir(nnunet_input_images) if f.endswith('_0000.nii.gz')]
        if len(preprocessed_files) == 0:
            logger.error("No files were successfully preprocessed!")
            return output_dir_final

        logger.info("Running nnUNetv2 prediction...")
        model_dir = f"/app/ingested_program/sample_code_submission/{self.dataset_id}/nnUNetTrainer_nnUNetPlans_{self.config}"

        if not os.path.exists(model_dir):
            logger.error(f"Model directory does not exist: {model_dir}")
            return output_dir_final

        try:
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False
            )

            predictor.initialize_from_trained_model_folder(
                model_dir,
                use_folds=(0, 1, 2, 3,4,),  # Still using 4 folds
                checkpoint_name='checkpoint_final.pth'
            )

            nnunet_image_files = []
            for f in os.listdir(nnunet_input_images):
                if f.endswith("_0000.nii.gz"):
                    base_id = f.replace("_0000.nii.gz", "")
                    file_triplet = [
                        os.path.join(nnunet_input_images, f"{base_id}_0000.nii.gz"),
                        os.path.join(nnunet_input_images, f"{base_id}_0001.nii.gz"),
                        os.path.join(nnunet_input_images, f"{base_id}_0002.nii.gz")
                    ]
                    nnunet_image_files.append(file_triplet)

            predictor.predict_from_files_sequential(
                nnunet_image_files,
                output_dir_nnunet,
                save_probabilities=False,
                overwrite=True,
                folder_with_segs_from_prev_stage=None
            )
        except Exception as e:
            logger.error(f"Error during nnUNet prediction: {str(e)}")
            return output_dir_final

        logger.info("Processing predictions...")
        nnunet_predictions = [f for f in os.listdir(output_dir_nnunet) if f.endswith('.nii.gz')]

        for fname in nnunet_predictions:
            try:
                patient_id = fname.replace(".nii.gz", "").replace("_0000", "")
                patient_id_upper = patient_id.upper()

                src_path = os.path.join(output_dir_nnunet, fname)
                temp_output_path = os.path.join(output_dir_final, f"{patient_id_upper}_resampled.nii.gz")

                self.resample_to_original_space(src_path, patient_id_upper, temp_output_path)

                # === POSTPROCESSING: keep largest connected component ===
                try:
                    prediction_img = sitk.ReadImage(temp_output_path)
                    prediction_array = sitk.GetArrayFromImage(prediction_img)
                    processed_array = keep_largest_component(prediction_array)
                    postprocessed_img = sitk.GetImageFromArray(processed_array)
                    postprocessed_img.CopyInformation(prediction_img)
                    sitk.WriteImage(postprocessed_img, temp_output_path)
                    logger.info(f"Postprocessing applied for {patient_id_upper}")
                except Exception as e:
                    logger.warning(f"Postprocessing failed for {patient_id_upper}: {str(e)}")

                final_output_path = os.path.join(output_dir_final, f"{patient_id_upper}.nii.gz")
                shutil.copy(temp_output_path, final_output_path)

                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

            except Exception as e:
                logger.error(f"Error processing prediction for {fname}: {str(e)}")
                continue

        final_segmentations = [f for f in os.listdir(output_dir_final) if f.endswith('.nii.gz')]
        logger.info(f"Final segmentations created: {len(final_segmentations)} files")

        self.predicted_segmentations = output_dir_final
        logger.info(f"Final segmentations saved to: {output_dir_final}")
        return output_dir_final
