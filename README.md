# 🌿 Dermora AI — Dermatological Computer Vision Prototype

[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2016%20%2B%20TypeScript-000000?logo=nextdotjs)](https://nextjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Styling-Tailwind%20CSS-38B2AC?logo=tailwindcss)](https://tailwindcss.com/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20REST-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python)](https://python.org/)
[![Status](https://img.shields.io/badge/Stage-Research%20Prototype-amber)](#-mandatory-research--educational-disclaimer)

**Dermora AI** is an advanced dermatological computer-vision research platform designed to evaluate multi-class skin lesion image classification using standardized International Skin Imaging Collaboration (ISIC) criteria.

---

## ⚠️ Mandatory Research & Educational Disclaimer

> **IMPORTANT:** Dermora AI is an investigative research and educational prototype. It is **NOT** a certified medical device, has not received FDA or CE-MDR clearance, and does **NOT** provide clinical medical diagnoses or treatment advice.
>
> Histopathological biopsy by a licensed dermatopathologist remains the gold standard for skin cancer diagnosis. Never defer or substitute professional medical care based on algorithmic outputs.

---

## 🎨 Official Visual Identity

The project features the **Dermora AI** brandmark—a neural network shield combined with a botanical balance motif in cyan, deep navy, and teal, symbolizing clinical precision and skin health protection.
- Stored as `image_bea0ca7.jpg` in the project root and served as `frontend/public/dermora-logo.jpg`.
- Rendered with strict aspect-ratio preservation across the navbar, home page, footer, and browser favicon.

---

## 🏗️ System Architecture

The modernized codebase separates UI and inference into dedicated services:

```
skin-cancer-detection-ai/
├── image_bea0ca7.jpg          # [Original] Dermora AI brandmark
├── backend/                   # FastAPI REST Inference Engine
│   ├── app/
│   │   ├── main.py            # API routes (/api/predict, /api/results/{id}, /api/model-info, /api/health)
│   │   ├── model/             # Deterministic feature analysis & quality checks
│   │   │   ├── pipeline.py    # Sharpness, contrast, color variegation, asymmetry
│   │   │   └── labels.py      # Standardized 7 ISIC lesion classes
│   │   ├── training/          # Dataset training blueprint (HAM10000 / ISIC)
│   │   │   └── train.py
│   │   └── schemas.py         # Type-safe Pydantic models
│   └── requirements.txt
├── frontend/                  # Next.js 16+ App Router, TypeScript, Tailwind CSS
│   ├── public/
│   │   └── dermora-logo.jpg   # Brand logo
│   ├── app/
│   │   ├── layout.tsx         # Navbar with logo, clinical alert banner, footer
│   │   ├── page.tsx           # Home: Hero, ABCDE framework, ISIC classes
│   │   ├── scan/page.tsx      # Drag-and-drop diagnostic image scanner
│   │   ├── results/[id]/page.tsx # Dedicated result view with ranked probabilities
│   │   ├── insights/page.tsx  # Dataset distribution & training roadmap
│   │   └── about/page.tsx     # Research mission & ethical disclaimers
│   └── package.json
└── skin_cancer_app/           # [Preserved] Legacy Streamlit prototype
```

---

## 🔬 Supported Lesion Taxonomy (ISIC / HAM10000)

Dermora AI models classifications across seven dermatological lesion types:

1. **MEL (Melanoma):** Malignant neoplasm of melanocytes *(High Risk)*
2. **NV (Melanocytic Nevus):** Common benign mole *(Low Risk)*
3. **BCC (Basal Cell Carcinoma):** Common non-melanoma skin cancer *(High Risk)*
4. **AKIEC (Actinic Keratosis):** Pre-cancerous sun-induced lesion *(Medium Risk)*
5. **BKL (Benign Keratosis):** Seborrheic keratosis, solar lentigo *(Low Risk)*
6. **DF (Dermatofibroma):** Benign fibrous nodule *(Low Risk)*
7. **VASC (Vascular Lesion):** Hemangioma, angiokeratoma *(Low Risk)*

---

## 🚀 Running the Project Locally

### 1. Start the FastAPI Backend
```bash
# From repository root:
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health Check: `http://127.0.0.1:8000/api/health`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### 2. Start the Next.js Frontend
```bash
# In a separate terminal, navigate to frontend:
cd frontend
npm install
npm run dev
```
- Open [http://localhost:3000](http://localhost:3000) in your browser.

### 3. [Optional] Legacy Streamlit App
The original Streamlit application remains preserved in `skin_cancer_app/`:
```bash
streamlit run skin_cancer_app/app.py
```

---

## 📊 Scientific Integrity & Model Provenance

- **No Fake Random Sampling:** All legacy synthetic Dirichlet sampling (`np.random.dirichlet`) and hardcoded 94.5% clinical claims have been excised.
- **Deterministic Baseline:** The current prototype extracts deterministic mathematical features (gradient sharpness variance, intensity contrast, color channel variegation, spatial asymmetry).
- **Training Blueprint:** A reproducible PyTorch training pipeline on the 10,015-image HAM10000 / ISIC dataset is staged in `backend/app/training/train.py`.

---

## 📜 Dataset Attribution and Licence

This project utilizes the **HAM10000** ("Human Against Machine with 10,000 training images") dataset for research benchmarking and model evaluation.

- **Citation:** Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161 (2018). [https://doi.org/10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)
- **Official Repository:** Harvard Dataverse, persistent link: [https://doi.org/10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)
- **Licence Terms:** The HAM10000 dataset is distributed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** licence, subject to the terms and stipulations specified by the original authors and repository depositors.
- **Downstream Usage Notice:** The non-commercial restriction of the source dataset may affect downstream use, redistribution, and deployment of the dataset and related project outputs. Prospective users and downstream developers are advised to independently review the original licence terms to assess compliance with their intended use case.
