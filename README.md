

# Drug Mechanism Prediction from Cell Images using AI

**Live App:** [https://ranjith445-bbbc021-moa-predictor.hf.space](https://ranjith445-bbbc021-moa-predictor.hf.space)

---

## What is this project about?

When scientists develop new medicines, they need to understand **how a drug affects cells**.
This is called the **Mechanism of Action (MoA)**.

To study this, they take **microscope images of human cells** after applying different drugs.

* Each drug changes the cells in a different way
* These changes can be seen in images
* But analyzing thousands of images manually is slow

This project builds an **AI system that can study these images and predict how a drug works**.

---

## What does this system do?

This system:

* Takes microscope images of cells
* Analyzes how the cells look after drug treatment
* Predicts the **type of effect (MoA)** of the drug
* Shows **which parts of the image influenced the decision**
* Generates a **report with the results**

---

## How does it work (simple explanation)

The system looks at the images in a step-by-step way:

### 1. Reads the cell images

Each image has 3 types of information:

* Cell nucleus (DNA)
* Cell structure (tubulin)
* Cell shape (actin)

---

### 2. Cleans and prepares the image

* Adjusts brightness and contrast
* Resizes the image
* Identifies important parts like the nucleus

---

### 3. Extracts useful patterns

The system looks for:

* Shape of cells
* Size and structure
* Intensity (brightness patterns)

---

### 4. Uses two types of AI together

**Image model (ResNet18)**

* Looks at the image patterns
* Learns visual features

**Data model (MLP)**

* Uses additional numerical information (synthetic omics data)

---

### 5. Combines both

* Both models share their understanding
* The system combines them
* Makes the final prediction

---

### 6. Predicts drug effect

It classifies the drug into one of these types:

* Microtubule stabilizers
* Microtubule destabilizers
* Aurora kinase inhibitors
* Actin disruptors

---

## What results does it give?

* Accuracy: 80.6%
* F1 Score: 0.867

This shows good performance for a research-level demonstration.

---

## What data was used?

* Dataset: BBBC021 (from Broad Institute)
* Images of breast cancer cells (MCF-7)

For this project:

* 372 image sites used
* 4 drug effect classes
* 972 training images
* 144 validation images

---

## What makes this project special?

### Combines multiple types of data

* Image data
* Additional numerical data

### Shows explanation

* Highlights which parts of the image influenced the result

### Generates reports

* Creates a downloadable PDF with full analysis

---

## Features of the application

* Select sample cell images
* Predict drug mechanism
* View confidence scores
* See highlighted important regions (GradCAM)
* Download PDF report

---

## How to use

1. Open the web app
2. Select an image from the list
3. Click “Run Prediction”
4. View the results and explanation
5. Download the report if needed

---

## Important Note

* This model is trained on a **limited dataset (Week 1 subset only)**
* Uses **synthetic omics data**, not real biological data

This project is meant for **learning and demonstration**.

---

## Limitations

* Only 4 drug classes used (out of 12 total)
* Not trained on full dataset
* Not validated for real-world drug research

---

## Future Improvements

* Use full dataset (14,400 images)
* Add real gene expression data
* Improve accuracy and robustness

---

## Disclaimer

This project is for educational and research purposes only.
It should not be used for medical or pharmaceutical decisions.

---

## One-Line Summary

An AI system that studies cell images to predict how a drug affects cells and explains its decision.

---

## Author

Built by M. Ranjith Kumar as a biomedical AI portfolio project.

---


