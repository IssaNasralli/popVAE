
# popVAE: Fine-Scale Population Estimation Using a Variational Autoencoder-Based Approach

This repository provides the official implementation and data preparation scripts for the paper:

**Fine-Scale Population Estimation Using a Variational Autoencoder-Based Approach**  
*Issa Nasralli, Imen Masmoudi, Hassen Drira, Mohamed Ali Hadj Taieb*  
📄 [Read the paper](./2025145777.pdf)

---

## 🧠 Overview

This repository supports our work on **popVAE**, a deep learning model for high-resolution population mapping using spatial contextual features extracted via a Variational Autoencoder (VAE). The codebase includes:

- Dataset extraction scripts using **Google Earth Engine**
- Preprocessing tools using **ArcPy**
- A VAE-based deep learning architecture implemented in **Python (TensorFlow 2.15/Keras)**
- Links to all necessary datasets

---

## 📁 Repository Structure

```bash
popVAE/
├── gee_scripts/             # JavaScript files to extract datasets from Google Earth Engine
├── ArcGIS_preprocessing/    # ArcPy-based preprocessing scripts for geospatial data
├── popVAE_model/            # Python code for the popVAE model, training, and inference
├── README.md                # This file
