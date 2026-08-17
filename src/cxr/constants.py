"""NIH ChestX-ray14 label set.

Order matches the CheXNet paper (Rajpurkar et al., 2017) so published
per-class AUCs can be compared directly.
"""

from __future__ import annotations

PATHOLOGIES: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CLASSES: int = len(PATHOLOGIES)
PATHOLOGY_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(PATHOLOGIES)}

# NIH also uses a space in this name. Both spellings map to the same class.
LABEL_ALIASES: dict[str, str] = {
    "Pleural Thickening": "Pleural_Thickening",
    "Pleural_thickening": "Pleural_Thickening",
    "No Finding": "No Finding",
}

# ImageNet mean/std. We replicate the grayscale CXR to 3 channels, so
# ImageNet-pretrained DenseNet-121 can be used without a custom first layer.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Official NIH download (Clinical Center). Each tar is extracted, resized,
# then deleted so a laptop disk can hold the full 112,120-image set.
NIH_IMAGE_TARS: list[tuple[str, str]] = [
    ("https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz", "images_001.tar.gz"),
    ("https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz", "images_002.tar.gz"),
    ("https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz", "images_003.tar.gz"),
    ("https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz", "images_004.tar.gz"),
    ("https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz", "images_005.tar.gz"),
    ("https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz", "images_006.tar.gz"),
    ("https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz", "images_007.tar.gz"),
    ("https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz", "images_008.tar.gz"),
    ("https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz", "images_009.tar.gz"),
    ("https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz", "images_010.tar.gz"),
    ("https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz", "images_011.tar.gz"),
    ("https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz", "images_012.tar.gz"),
]

# Metadata mirrors. First successful download wins.
NIH_METADATA_URLS: dict[str, list[str]] = {
    "Data_Entry_2017.csv": [
        "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/Data_Entry_2017.csv",
        "https://raw.githubusercontent.com/zoogzog/chexnet/master/dataset/Data_Entry_2017.csv",
        "https://raw.githubusercontent.com/brucechou1983/CheXNet-Keras/master/data/Data_Entry_2017.csv",
    ],
    "BBox_List_2017.csv": [
        "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/BBox_List_2017.csv",
        "https://raw.githubusercontent.com/zoogzog/chexnet/master/dataset/BBox_List_2017.csv",
    ],
    "train_val_list.txt": [
        "https://raw.githubusercontent.com/zoogzog/chexnet/master/dataset/train_val_list.txt",
        "https://raw.githubusercontent.com/brucechou1983/CheXNet-Keras/master/data/train_val_list.txt",
        "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/train_val_list.txt",
    ],
    "test_list.txt": [
        "https://raw.githubusercontent.com/zoogzog/chexnet/master/dataset/test_list.txt",
        "https://raw.githubusercontent.com/brucechou1983/CheXNet-Keras/master/data/test_list.txt",
        "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/test_list.txt",
    ],
}

# Short clinical phrases used by the report composer.
FINDING_LANGUAGE: dict[str, dict[str, str]] = {
    "Atelectasis": {
        "mild": "linear opacities suggesting subsegmental atelectasis",
        "moderate": "focal atelectasis with volume loss",
        "severe": "lobar collapse with marked volume loss",
    },
    "Cardiomegaly": {
        "mild": "the cardiomediastinal silhouette is mildly enlarged",
        "moderate": "moderate cardiomegaly is present",
        "severe": "marked cardiomegaly with an increased cardiothoracic ratio",
    },
    "Effusion": {
        "mild": "blunting of the costophrenic angle suggesting a small pleural effusion",
        "moderate": "a moderate pleural effusion is present",
        "severe": "a large pleural effusion with adjacent compressive atelectasis",
    },
    "Infiltration": {
        "mild": "faint interstitial opacities",
        "moderate": "patchy airspace opacities consistent with infiltrate",
        "severe": "extensive airspace disease / infiltrate",
    },
    "Mass": {
        "mild": "a focal opacity that may represent a small mass",
        "moderate": "a discrete pulmonary mass is identified",
        "severe": "a large pulmonary mass is present",
    },
    "Nodule": {
        "mild": "a faint pulmonary nodule is suggested",
        "moderate": "a discrete pulmonary nodule is present",
        "severe": "a dominant pulmonary nodule / mass-like nodule is present",
    },
    "Pneumonia": {
        "mild": "subtle airspace opacity that may represent early pneumonia",
        "moderate": "focal consolidation compatible with pneumonia",
        "severe": "multilobar consolidation compatible with pneumonia",
    },
    "Pneumothorax": {
        "mild": "a small apical pneumothorax is suggested",
        "moderate": "a moderate pneumothorax is present",
        "severe": "a large pneumothorax with possible tension features",
    },
    "Consolidation": {
        "mild": "ill-defined consolidation",
        "moderate": "focal lobar consolidation",
        "severe": "dense multilobar consolidation",
    },
    "Edema": {
        "mild": "cephalization of pulmonary vasculature suggesting mild edema",
        "moderate": "interstitial pulmonary edema with Kerley lines",
        "severe": "alveolar pulmonary edema with bat-wing opacities",
    },
    "Emphysema": {
        "mild": "mild pulmonary hyperinflation suggesting emphysema",
        "moderate": "hyperinflation and attenuated peripheral markings of emphysema",
        "severe": "severe emphysema with marked hyperinflation and bullous change",
    },
    "Fibrosis": {
        "mild": "subtle reticular opacities suggesting fibrosis",
        "moderate": "reticular interstitial markings consistent with fibrosis",
        "severe": "coarse reticular fibrosis with architectural distortion",
    },
    "Pleural_Thickening": {
        "mild": "minimal pleural thickening",
        "moderate": "focal pleural thickening",
        "severe": "extensive pleural thickening",
    },
    "Hernia": {
        "mild": "a small hiatal hernia is suggested",
        "moderate": "a hiatal hernia is present",
        "severe": "a large hiatal / diaphragmatic hernia is present",
    },
}
