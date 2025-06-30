# Functional Connectivity Decoding of P300 Responses in Brain-Computer Interfaces Using Surrogate-Enhanced Weighted Phase Lag Index

![Graphical Abstract](abstract/Fig_Graphical_Abstract_cut.png)

Here we demonstrate how statistically significant fronto-parietal functional connectivity, extracted using surrogate-thresholded wPLI, enable accurate cross-subject and cross-group classification of P300 responses in a BCI paradigm. The approach generalizes across healthy and clinical populations, highlighting the robustness of connectivity-based features for P300 decoding

---

## Abstract

The ultimate objective of Brain-Computer Interfaces (BCIs) is to translate brain activity into computer-interpretable commands. In this context, feature extraction often focuses on the temporal properties of electroencephalography (EEG) signals, which are significantly influenced by inter-individual variability and clinical conditions. Exploiting the interconnected nature of the brain, this study characterizes the functional connectivity networks involved in generating P300 event-related potentials (ERPs) during BCI paradigms.

The dataset included eight participants who viewed a laptop screen displaying six images flashing randomly to control appliances via BCI. When the image the subject was fixating on appeared, a P300 ERP was elicited. For feature extraction, we computed the weighted Phase Lag Index (wPLI) and other network metrics, retaining only connections exceeding 95% of surrogate values. These significant links were used to classify whether a run signal contained a P300 or not.

We analyzed differences between four participants with disabilities and four healthy students. Two classification algorithms were applied: Support Vector Classification (SVC) and Bayesian Linear Discriminant Analysis (BLDA), alongside leave-one-subject-out (LOSO) and leave-one-group-out (LOGO) cross-validation. Results revealed a stable pattern of fronto-parietal wPLI hypersynchronization during P300 events. The SVC classifier achieved 89.9% accuracy in LOSO-CV. Training on the control group achieved 84.8% accuracy in identifying P300 in the clinical group, and vice versa reached 79.9%.

These results support previous findings that P300 trials show increased connectivity and demonstrate the potential for generalizing connectivity-based features across subjects and clinical populations.

---

## Repository structure

- `abstract/` — Contains graphical abstract figure.
- `data/` — Data files (large files may be linked externally).
- `scripts/` — Analysis and processing scripts.

---

## External data

Due to file size limitations, some large data files are available on Google Drive:

[Google Drive Folder](https://drive.google.com/drive/folders/1_tWZ4cRi73jq1s3Sq_2EUIxUapawSEzT?usp=sharing)

---

## Usage

(Here you can add instructions on how to run the scripts, dependencies, etc.)

---

## License

(Add license info here)



