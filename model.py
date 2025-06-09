# ==================== MAMA-MIA CHALLENGE FPIXEL SUBMISSION ====================
# Primary Tumour Segmentation (Task 1)

import os
import numpy as np
import shutil
import SimpleITK as sitk
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import logging

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                
                if len(phase_files) < 2:
                    logger.warning(f"Skipping {patient_id}: only {len(phase_files)} phases available")
                    continue
                
                # Use only first post-contrast (index 1)
                image_path = phase_files[1]
                logger.info(f"Reading image: {image_path}")
                
                image = sitk.ReadImage(image_path, sitk.sitkFloat32)
                original_size = image.GetSize()
                original_spacing = image.GetSpacing()
                logger.info(f"Original image - Size: {original_size}, Spacing: {original_spacing}")
                
                # Get original statistics
                array = sitk.GetArrayFromImage(image)
                logger.info(f"Original image stats - Min: {array.min():.3f}, Max: {array.max():.3f}, Mean: {array.mean():.3f}")
                
                image = self.make_isotropic(image, interpolator=sitk.sitkBSpline)
                new_size = image.GetSize()
                new_spacing = image.GetSpacing()
                logger.info(f"After resampling - Size: {new_size}, Spacing: {new_spacing}")
                
                image = self.z_score_normalize(image)
                
                # Get normalized statistics
                norm_array = sitk.GetArrayFromImage(image)
                logger.info(f"After normalization - Min: {norm_array.min():.3f}, Max: {norm_array.max():.3f}, Mean: {norm_array.mean():.3f}")
                
                output_filename = f"{patient_id}_0000.nii.gz"
                output_path = os.path.join(preprocessed_output_dir, output_filename)
                sitk.WriteImage(image, output_path)
                
                logger.info(f"Successfully preprocessed {patient_id}")
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

    @staticmethod
    def z_score_normalize(image_sitk):
        array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
        mean = np.mean(array)
        std = np.std(array)
        
        # Avoid division by zero
        if std == 0:
            std = 1.0
            logger.warning("Standard deviation is 0, using 1.0 instead")
        
        norm_array = (array - mean) / std
        norm_image = sitk.GetImageFromArray(norm_array)
        norm_image.CopyInformation(image_sitk)
        return norm_image

    def resample_to_original_space(self, prediction_path, patient_id, output_path):
        """Resample prediction from isotropic space back to original image space"""
        try:
            # Load the prediction (in isotropic space)
            prediction = sitk.ReadImage(prediction_path)
            pred_array = sitk.GetArrayFromImage(prediction)
            logger.info(f"Prediction stats for {patient_id} - Shape: {pred_array.shape}, Unique values: {np.unique(pred_array)}")
            
            # Load the original image (first post-contrast) to get reference space
            phase_files = self.dataset.get_dce_mri_path_list(patient_id)
            original_image = sitk.ReadImage(phase_files[1])
            
            # Resample prediction to match original image space
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(original_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for segmentation
            resampler.SetDefaultPixelValue(0)
            
            resampled_prediction = resampler.Execute(prediction)
            
            # Check resampled prediction
            resampled_array = sitk.GetArrayFromImage(resampled_prediction)
            logger.info(f"Resampled prediction for {patient_id} - Shape: {resampled_array.shape}, Unique values: {np.unique(resampled_array)}")
            
            sitk.WriteImage(resampled_prediction, output_path)
            return output_path
            
        except Exception as e:
            logger.error(f"Error resampling prediction for {patient_id}: {str(e)}")
            # Create empty segmentation as fallback
            phase_files = self.dataset.get_dce_mri_path_list(patient_id)
            original_image = sitk.ReadImage(phase_files[1])
            empty_seg = sitk.Image(original_image.GetSize(), sitk.sitkUInt8)
            empty_seg.CopyInformation(original_image)
            sitk.WriteImage(empty_seg, output_path)
            return output_path

    def apply_breast_mask(self, seg_path, patient_id, output_path):
        """Apply breast coordinate mask to segmentation"""
        try:
            segmentation = sitk.ReadImage(seg_path)
            segmentation_array = sitk.GetArrayFromImage(segmentation)

            patient_info = self.dataset.read_json_file(patient_id)
            coords = patient_info["primary_lesion"]["breast_coordinates"]
            x_min, x_max = coords["x_min"], coords["x_max"]
            y_min, y_max = coords["y_min"], coords["y_max"]
            z_min, z_max = coords["z_min"], coords["z_max"]
            
            logger.info(f"Breast coordinates for {patient_id}: x[{x_min}:{x_max}], y[{y_min}:{y_max}], z[{z_min}:{z_max}]")

            # Create masked segmentation
            masked_segmentation = np.zeros_like(segmentation_array)
            
            # Ensure coordinates are within bounds
            z_max = min(z_max, segmentation_array.shape[0])
            y_max = min(y_max, segmentation_array.shape[1])
            x_max = min(x_max, segmentation_array.shape[2])
            
            z_min = max(z_min, 0)
            y_min = max(y_min, 0)
            x_min = max(x_min, 0)
            
            # Apply mask
            masked_segmentation[z_min:z_max, y_min:y_max, x_min:x_max] = \
                 segmentation_array[z_min:z_max, y_min:y_max, x_min:x_max]

            # Check if mask removed everything
            before_voxels = np.sum(segmentation_array > 0)
            after_voxels = np.sum(masked_segmentation > 0)
            logger.info(f"Masking {patient_id}: {before_voxels} -> {after_voxels} positive voxels")

            masked_seg_image = sitk.GetImageFromArray(masked_segmentation)
            masked_seg_image.CopyInformation(segmentation)
            sitk.WriteImage(masked_seg_image, output_path)
            
        except Exception as e:
            logger.error(f"Error applying breast mask for {patient_id}: {str(e)}")
            # Copy original segmentation if masking fails
            segmentation = sitk.ReadImage(seg_path)
            sitk.WriteImage(segmentation, output_path)



    def predict_segmentation(self, output_dir):
        logger.info("Setting up directories...")

        # Input/output directories
        nnunet_input_images = os.path.join(output_dir, 'nnunet_input_images')
        os.makedirs(nnunet_input_images, exist_ok=True)

        output_dir_nnunet = os.path.join(output_dir, 'nnunet_seg')
        os.makedirs(output_dir_nnunet, exist_ok=True)

        output_dir_final = os.path.join(output_dir, 'pred_segmentations')
        os.makedirs(output_dir_final, exist_ok=True)

        # Step 1: Preprocess to isotropic space
        logger.info("Starting preprocessing...")
        self.preprocess_images(nnunet_input_images)

        # Check if any files were preprocessed
        preprocessed_files = [f for f in os.listdir(nnunet_input_images) if f.endswith('.nii.gz')]
        logger.info(f"Found {len(preprocessed_files)} preprocessed files: {preprocessed_files}")
        
        if len(preprocessed_files) == 0:
            logger.error("No files were successfully preprocessed!")
            return output_dir_final

        # Step 2: Run nnUNet prediction
        logger.info("Running nnUNetv2 prediction...")
        
        # Check if model directory exists
        model_dir = f"/app/ingested_program/sample_code_submission/{self.dataset_id}/nnUNetTrainer_nnUNetPlans_{self.config}"
        logger.info(f"Model directory: {model_dir}")
        
        if not os.path.exists(model_dir):
            logger.error(f"Model directory does not exist: {model_dir}")
            # List available directories for debugging
            base_dir = "/app/ingested_program/sample_code_submission"
            if os.path.exists(base_dir):
                logger.info(f"Available directories in {base_dir}: {os.listdir(base_dir)}")
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
                use_folds=(0,1,2,3,4),
                checkpoint_name='checkpoint_final.pth'
            )
            logger.info("Model loaded successfully")

            # Predict
            nnunet_image_files = [
                [os.path.join(nnunet_input_images, f)]
                for f in os.listdir(nnunet_input_images) if f.endswith(".nii.gz")
            ]
            
            logger.info(f"Predicting on {len(nnunet_image_files)} files")

            predictor.predict_from_files_sequential(
                nnunet_image_files,
                output_dir_nnunet,
                save_probabilities=False,
                overwrite=True,
                folder_with_segs_from_prev_stage=None
            )
            logger.info("Prediction succeeded.")
            
        except Exception as e:
            logger.error(f"Error during nnUNet prediction: {str(e)}")
            return output_dir_final

        # Step 3: Process predictions
        logger.info("Processing predictions...")
        nnunet_predictions = [f for f in os.listdir(output_dir_nnunet) if f.endswith('.nii.gz')]
        logger.info(f"Found {len(nnunet_predictions)} nnUNet predictions: {nnunet_predictions}")
        
        for fname in nnunet_predictions:
            try:
                patient_id = fname.replace(".nii.gz", "").replace("_0000", "")
                patient_id_upper = patient_id.upper()
                
                src_path = os.path.join(output_dir_nnunet, fname)
                temp_output_path = os.path.join(output_dir_final, f"{patient_id_upper}_resampled.nii.gz")

                logger.info(f"Processing prediction for patient: {patient_id}")
                
                # Resample to original space
                self.resample_to_original_space(src_path, patient_id_upper, temp_output_path)

                # Apply breast mask
                final_output_path = os.path.join(output_dir_final, f"{patient_id_upper}.nii.gz")
                shutil.copy(temp_output_path, final_output_path)
                
                # Clean up temporary file
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                    
            except Exception as e:
                logger.error(f"Error processing prediction for {fname}: {str(e)}")
                continue

        # Final verification
        final_segmentations = [f for f in os.listdir(output_dir_final) if f.endswith('.nii.gz')]
        logger.info(f"Final segmentations created: {len(final_segmentations)} files")

        self.predicted_segmentations = output_dir_final
        logger.info(f"Final segmentations saved to: {output_dir_final}")
        return output_dir_final
