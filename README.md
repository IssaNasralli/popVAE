# Population Estimation Using Deep Learning and Geospatial Data
This repository provides :

1) the official implementation and data preparation scripts for our IJCNN paper,

**Fine-Scale Population Estimation Using a Variational Autoencoder-Based Approach**  
*Issa Nasralli, Imen Masmoudi, Hassen Drira, Mohamed Ali Hadj Taieb*  

2) the official implementation and data preparation scripts for our future journal paper which extend our IJCNN paper,

**Three-Branch Hybrid Model for Fine-Grained Population Estimation with Adaptive Spatial Context Integration**
*Issa Nasralli, Imen Masmoudi, Hassen Drira, Mohamed Ali Hadj Taieb*  

3) the final training raster dataset

4) the population datasets
---

## 🧠 Overview

This repository supports our work on **popVAE**, a deep learning model for high-resolution population mapping using spatial contextual features extracted via a Variational Autoencoder (VAE). The codebase includes:

- Dataset extraction scripts using **Google Earth Engine**
- Preprocessing tools using **ArcPy**
- The two deep learning architectures implemented in **Python (TensorFlow 2.15/Keras)**
- Links to all necessary datasets

---

## 📂 Final Training Raster Dataset

The final training raster dataset for **Tunisia** generated is available here:  
👉 [Download Tunisia Final Raster (Google Drive)](https://drive.google.com/file/d/12YaLwfOp-IPpgUMciMzb_lOR_eK4aL5B/view?usp=sharing)

## 📂 Population  Dataset

The Worldpop population raster dataset for **Tunisia** (2020) is available here:  
👉 [Download (Google Drive)](https://drive.google.com/file/d/144qTJMNqwMi6JjsT-KorP9HB4IgxaWe5/view?usp=sharing)

The GPWv4 population raster dataset for **Tunisia** (2020) is available here:  
👉 [Download (Google Drive)](https://drive.google.com/file/d/1HrbtDfSGfP6dj6BfA91Kp_74GM1ixNVV/view?usp=sharing)

The Landscan population raster dataset for **Tunisia** (2020) is available here:  
👉 [Download (Google Drive)](https://drive.google.com/file/d/1PRqruoDi6GpFlOaR--N2v_h0sjl88jMz/view?usp=sharing)


The INS population tab ular dataset for **Tunisia** (2020) is available in this repositry (ins_population.csv)  




---

## 📁 Repository Structure

```bash
popVAE/
├── gee_scripts/             # JavaScript files to extract datasets from Google Earth Engine
├── ArcGIS_preprocessing/    # ArcPy-based preprocessing scripts for geospatial data
├── popVAE_model/            # Python code for the popVAE model, training, and inference
├── popVAT_model/            # Python code for the popVAT model, training, and inference
├── ins_population.csv       # The tabular population of INS 
├── README.md                # This file
