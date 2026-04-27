Multi-Modal Fluorescence Microscopy Analysis Pipeline

Overview
This project predicts the Mechanism of Action (MoA) of drugs from fluorescence microscopy images of human breast cancer cells (MCF-7) using a multi-modal deep learning pipeline trained on the BBBC021 dataset from the Broad Institute. The system combines a ResNet18 image branch with a synthetic omics MLP branch, fused together to classify drug mechanisms from raw cell imaging data.

Dataset
BBBC021 (Broad Bioimage Benchmark Collection) — fluorescence microscopy images of MCF-7 human breast cancer cells treated with 103 chemical compounds across 12 mechanism of action classes. This project uses Week1 data (372 sites, 4 MoA classes, compound-level train/val split).
MoA ClassSitesMicrotubule stabilizers216Aurora kinase inhibitors72Actin disruptors48Microtubule destabilizers36Total372

Pipeline

Raw .tif fluorescence images (DAPI, Tubulin, Actin channels)
Channel-wise percentile normalization
Resize to 256x256
U-Net nuclei segmentation using Otsu pseudo masks
Morphological and intensity feature extraction
Multi-modal fusion (ResNet18 CNN + MLP omics branch)
MoA classification (4 classes)
GradCAM explainability visualization
PDF report generation
Streamlit deployment on Hugging Face Spaces


Model Architecture
BranchArchitectureOutputImage branchResNet18 pretrained (ImageNet) + FC256-dim embeddingOmics branchMLP (64-dim synthetic omics)128-dim embeddingFusionConcat(384) → FC(256) → FC(128) → FC(4)MoA class

Performance
MetricValueValidation accuracy80.6%Weighted F1 score0.867Training sites324Validation sites48Train/val split strategyCompound-levelTraining images972 (324 sites × 3 channels)Validation images144 (48 sites × 3 channels)

App Features

Preloaded sample images from BBBC021 dataset
MoA prediction with confidence scores per class
GradCAM visualization showing which image regions drove the prediction
Downloadable PDF report with full analysis
Deployed on Hugging Face Spaces via Docker


Technologies
CategoryToolsDeep learningPyTorch, torchvision (ResNet18)Image processingOpenCV, scikit-image, tifffileFeature extractionscikit-learn, scipy, NumPyDeploymentStreamlit, Docker, Hugging Face SpacesReportingReportLab (PDF generation)Datapandas, matplotlib

Deployment
https://ranjith445-bbbc021-moa-predictor.hf.space

How to Use

Open the app at the Hugging Face Spaces link above
Select an imaging site from the dropdown in the sidebar
Click Run Prediction
View fluorescence channels, predicted MoA, confidence scores and GradCAM
Download the PDF report


Limitations

Trained on Week1 subset only (372 sites, 4 MoA classes)
Synthetic omics vectors used — no real gene expression data
Pipeline designed to scale to full 10-week dataset (14,400 sites, 12 MoA classes)
Not validated for clinical use


Dataset Citation
Ljosa V, Sokolnicki KL, Bhagavatula P, et al. Annotated high-throughput microscopy image sets for validation. Nature Methods, 2012.

Author
Built by M. Ranjith Kumar as a biomedical AI portfolio project demonstrating end-to-end fluorescence microscopy analysis for drug mechanism of action prediction.
