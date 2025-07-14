# 🌸 MAMA-MIA Challenge 2025: Primary Tumour Segmentation in DCE-MRI

**Authors:** Beyza Zayim¹, Aissiou Ikram², Boukhiar Naima³  
¹ Université de Bourgogne, Dijon 21000, France  
² Algeria  
³ University of Algiers 1 Ben Youcef Ben Khedda, Algeria

## 🧬 Overview

This repository contains our submission for the MICCAI 2025 MAMA-MIA Challenge, focusing on primary tumor segmentation in Dynamic Contrast-Enhanced MRI (DCE-MRI) breast cancer data. Our approach leverages the nnU-Net framework with a selective training strategy based on image quality and center-specific variability.

## Key Findings

- **Data Quality Matters**: Including low-quality ISPY scans impaired segmentation performance, even with advanced preprocessing
- **Center-Specific Strategy**: Training on high-quality DUKE and NACT data with early-phase images (0000–0002) yielded more robust results
- **Best Performance**: Achieved validation Dice score of 0.72 using multi-phase (phases 1-3) DUKE+NACT data
- **Postprocessing**: Keeping only the largest connected component significantly improved results

## 📁 Repository Structure

```
.
├── sample_code_submission/
│   ├── model.py                    # Main model implementation
│   ├── Dataset105_full_image/      # nnU-Net model directory
│   │   └── nnUNetTrainer_nnUNetPlans_3d_fullres/
│   │       ├── fold_0/
│   │       ├── fold_1/
│   │       ├── fold_2/
│   │       ├── fold_3/
│   │       └── fold_4/
└── ReadMe.md
```

## ⚙️ Method

### Pipeline Overview

1. **Dataset Preparation**: Multi-center DCE-MRI data from DUKE, NACT, ISPY1, and ISPY2 
2. **Preprocessing**: Isotropic resampling (1mm³) and Z-score normalization
3. **Model Training**: nnU-Net 3D full-resolution with 5-fold cross-validation
4. **Postprocessing**: Largest connected component filtering

### Key Technical Details

- **Model**: nnU-Net 3D full-resolution configuration
- **Input**: 3 temporal phases (0000-0002) for optimal performance
- **Training Data**: Selective use of high-quality DUKE (247 cases) and NACT (64 cases) data
- **Optimization**: Adam optimizer (lr=1e-4, weight_decay=3e-5)
- **Hardware**: NVIDIA A100 80GB GPU
- **Framework**: PyTorch 2.6.0, CUDA 11.8

## 💻 Installation

### Prerequisites

```bash
# Create conda environment
conda create -n mama-mia python=3.9
conda activate mama-mia

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install nnU-Net v2
pip install nnunetv2
```

## Usage

### Environment Setup

```bash
export nnUNet_raw="/path/to/your/nnUNet_raw"
export nnUNet_preprocessed="/path/to/your/nnUNet_preprocessed"
export nnUNet_results="/path/to/your/nnUNet_results"
```

### Running Inference

```python
from model import Model

# Initialize model
model = Model(dataset=your_dataset, 
              dataset_id="Dataset105_full_image", 
              config="3d_fullres")

# Run prediction
output_dir = model.predict_segmentation("/path/to/output")
```

### Model Features

- **Automatic Preprocessing**: Isotropic resampling and normalization
- **Multi-phase Support**: Handles 3 temporal DCE phases
- **Breast Region Masking**: Uses provided breast coordinates for focused segmentation
- **Robust Postprocessing**: Largest connected component filtering
- **Error Handling**: Comprehensive logging and fallback mechanisms

## 📊 Results

### Performance Metrics

| Experiment | Data | Phases | Validation Dice | DUKE_001 | ISPY1_1183 | ISPY2_332 | NACT_64 |
|------------|------|---------|-----------------|----------|-------------|-----------|---------|
| Final Model | DUKE+NACT | 3 (1-3) | **0.72** | 0.9394 | 0.7640 | 0.8967 | 0.9580 |
| Single Phase | DUKE+NACT | 1 (phase2) | 0.62 | 0.8625 | 0.7196 | 0.8111 | 0.9514 |
| All Centers | 1200 cases | 1 (phase2) | 0.45 | 0.8894 | 0.6739 | 0.5227 | 0.9334 |

### Key Insights

1. **Quality > Quantity**: Selective high-quality data outperformed larger, mixed-quality datasets
2. **Multi-phase Benefits**: Using 3 temporal phases improved generalization
3. **Center Variability**: DUKE and NACT data showed superior consistency compared to ISPY datasets
4. **Postprocessing Impact**: Largest connected component filtering eliminated false positives

## Technical Contributions

- **Quality-Aware Training**: Demonstrated that selective, high-quality data outperforms larger mixed datasets
- **Multi-phase Integration**: Showed benefits of temporal information in DCE-MRI segmentation
- **Robust Pipeline**: Implemented comprehensive error handling and fallback mechanisms
- **Postprocessing Innovation**: Applied connected component analysis for improved segmentation

**MAMA-MIA Dataset:**
```bibtex
@article{garrucho2025,
  title={A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations},
  author={Garrucho, Lidia and Kushibar, Kaisar and Reidel, Claire-Anne and Joshi, Smriti and Osuala, Richard and Tsirikoglou, Apostolia and Bobowicz, Maciej and Riego, Javier del and Catanese, Alessandro and Gwoździewicz, Katarzyna and Cosaka, Maria-Laura and Abo-Elhoda, Pasant M and Tantawy, Sara W and Sakrana, Shorouq S and Shawky-Abdelfatah, Norhan O and Salem, Amr Muhammad Abdo and Kozana, Androniki and Divjak, Eugen and Ivanac, Gordana and Nikiforaki, Katerina and Klontzas, Michail E and García-Dosdá, Rosa and Gulsun-Akpinar, Meltem and Lafcı, Oğuz and Mann, Ritse and Martín-Isla, Carlos and Prior, Fred and Marias, Kostas and Starmans, Martijn P A and Strand, Fredrik and Díaz, Oliver and Igual, Laura and Lekadir, Karim},
  journal = {Scientific Data},
  year = {2025},
  doi = {10.1038/s41597-025-04707-4},
  pages = {453},
  number = {1},
  volume = {12}
}
```

## Acknowledgments

This work was supported by the MAMA-MIA Challenge 2025. We thank the challenge organizers, data contributors, and the broader medical imaging community for their open-source tools and resources.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Beyza Zayim**: Université de Bourgogne, Dijon, France
- **Aissiou Ikram**: Algeria
- **Naima Boukhiar**: University of Algiers 1 Ben Youcef Ben Khedda, Algeria

For questions about the implementation, please open an issue in this repository.



