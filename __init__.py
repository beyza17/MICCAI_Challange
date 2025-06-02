class Preprocessed:
    def __init__(self, dataset, target_spacing=1.0):
        self.dataset = dataset
        self.target_spacing = target_spacing

    def preprocess_all(self, out_images_folder):
        os.makedirs(out_images_folder, exist_ok=True)

        patient_ids = self.dataset.get_patient_id_list()
        for patient_id in patient_ids:
            print(f"Processing {patient_id}...")

            # Get all MRI phases for this patient
            phase_files = self.dataset.get_dce_mri_path_list(patient_id)

            for phase_idx, phase_file in enumerate(phase_files):
                # Read image from file path
                image = sitk.ReadImage(phase_file, sitk.sitkFloat32)

                # Make isotropic
                image = self.make_isotropic(image, interpolator=sitk.sitkBSpline)

                # Normalize
                image = self.z_score_normalize(image)

                # Save preprocessed image
                output_filename = f"{patient_id}_000{phase_idx}.nii.gz"
                sitk.WriteImage(image, os.path.join(out_images_folder, output_filename))

        print("✅ Preprocessing complete.")
        return out_images_folder

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
        norm_array = (array - mean) / (std)
        norm_image = sitk.GetImageFromArray(norm_array)
        norm_image.CopyInformation(image_sitk)
        return norm_image
