import os
import SimpleITK as sitk
import numpy as np
import glob


class Preprocessed:
    def __init__(self, target_spacing=1.0):
        self.target_spacing = target_spacing

    def preprocess_all(self, images_folder, segmentations_folder, out_images_folder, out_segmentations_folder):
        os.makedirs(out_images_folder, exist_ok=True)
        os.makedirs(out_segmentations_folder, exist_ok=True)

        patient_ids = sorted(os.listdir(images_folder))
        for patient_id in patient_ids:
            patient_path = os.path.join(images_folder, patient_id)
            if not os.path.isdir(patient_path):
                continue
            print(f"Processing {patient_id}...")

            # Find all available phases
            phase_files = sorted(glob.glob(os.path.join(patient_path, f"{patient_id}_000*.nii.gz")))
            for phase_file in phase_files:
                phase = self.extract_phase_number(phase_file)
                image = self.read_mri_phase_from_patient_id(images_folder, patient_id, phase)
                image = self.make_isotropic(image, interpolator=sitk.sitkBSpline)
                image = self.z_score_normalize(image)
                output_filename = f"{patient_id}_000{phase}.nii.gz"
                sitk.WriteImage(image, os.path.join(out_images_folder, output_filename))

            # Process segmentation
            if os.path.exists(os.path.join(segmentations_folder, f"{patient_id}.nii.gz")):
                mask = self.read_segmentation_from_patient_id(segmentations_folder, patient_id)
                mask = self.make_isotropic(mask, interpolator=sitk.sitkNearestNeighbor)
                sitk.WriteImage(mask, os.path.join(out_segmentations_folder, f"{patient_id}.nii.gz"))

        print("✅ Preprocessing complete.")
        return out_images_folder, out_segmentations_folder


    @staticmethod
    def make_isotropic(image_sitk, target_spacing=1.0, interpolator=sitk.sitkBSpline):
        spacing = image_sitk.GetSpacing()
        size = image_sitk.GetSize()

        new_spacing = [target_spacing] * 3
        new_size = [
            int(round(size[i] * spacing[i] / new_spacing[i]))
            for i in range(3)
        ]

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

    def z_score_normalize(self, image_sitk):
        array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
        mean = np.mean(array)
        std = np.std(array)
        norm_array = (array - mean) / (std + 1e-5)
        norm_image = sitk.GetImageFromArray(norm_array)
        norm_image.CopyInformation(image_sitk)
        return norm_image
 
    def extract_phase_number(self, file_path):
        # Extracts the number from 'patient_000{n}.nii.gz'
        filename = os.path.basename(file_path)
        phase_str = filename.split("_000")[-1].replace(".nii.gz", "")
        return int(phase_str)

