from typing import Dict, Any, List

ISIC_CLASSES: List[Dict[str, Any]] = [
    {
        "code": "MEL",
        "name": "Melanoma",
        "risk_level": "High Risk",
        "description": "Malignant neoplasm of melanocytes. Requires prompt specialist dermatological evaluation and biopsy."
    },
    {
        "code": "NV",
        "name": "Melanocytic Nevus",
        "risk_level": "Low Risk",
        "description": "Benign proliferation of melanocytes (common mole). Routine monitoring for changes is advised."
    },
    {
        "code": "BCC",
        "name": "Basal Cell Carcinoma",
        "risk_level": "High Risk",
        "description": "Most common non-melanoma skin cancer. Locally invasive, requires professional dermatological management."
    },
    {
        "code": "AKIEC",
        "name": "Actinic Keratosis",
        "risk_level": "Medium Risk",
        "description": "Pre-cancerous sun-induced lesion or intraepithelial carcinoma. Deserves medical assessment and cryotherapy/treatment."
    },
    {
        "code": "BKL",
        "name": "Benign Keratosis",
        "risk_level": "Low Risk",
        "description": "Benign epidermal growth (seborrheic keratosis, solar lentigo). Generally harmless."
    },
    {
        "code": "DF",
        "name": "Dermatofibroma",
        "risk_level": "Low Risk",
        "description": "Benign fibrous nodule, typically found on extremities. Usually requires no intervention unless symptomatic."
    },
    {
        "code": "VASC",
        "name": "Vascular Lesion",
        "risk_level": "Low Risk",
        "description": "Benign vascular malformations or hemangiomas. Generally low concern unless bleeding or changing rapidly."
    }
]

CLASS_BY_CODE = {c["code"]: c for c in ISIC_CLASSES}
