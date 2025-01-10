# Network Intrusion Detection System using Machine Learning

## Overview
This project implements a Network Intrusion Detection System (NIDS) using three different machine learning approaches: Convolutional Neural Networks (CNN), Random Forest, and a Hierarchical Support Vector Machine (SVM). The system is trained and tested on the CICIDS 2017 dataset, which contains various types of network attacks and normal traffic patterns.

## Model Architecture

### 1. Hierarchical SVM
- Level 1: Binary classification (Benign vs Attack)
- Level 2: Attack group classification (DoS/DDoS, Brute Force, Reconnaissance)
- Level 3: Specific attack classification within each group
  - DoS/DDoS: GoldenEye, Hulk, Slowhttptest, Slowloris, DDoS
  - Brute Force: FTP-Patator, SSH-Patator, Heartbleed, Bot
  - Reconnaissance: PortScan, Infiltration

### 2. Random Forest
- Multiple configurations tested with different parameters:
  - Number of estimators: 100, 150, 200
  - Criterion: Entropy and Gini
- Feature scaling using StandardScaler
- SMOTE for handling class imbalance

### 3. CNN
- 5 Convolutional layers with batch normalization
- Dropout layers for regularization
- Dense layers for final classification
- Early stopping and learning rate scheduling

## Project Structure
```
.
├── data/
│   ├── test_features.parquet
│   └── train_features.parquet
├── weights/
│   ├── cnn_model_weights.pkl
│   ├── level_[1-3]_model.pkl
│   ├── new_cnn_weights.pkl
│   └── various_rf_models.pkl
├── testing.py
├── training.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DEVeloper1006/network-intrusion-detection.git
cd network-intrusion-detection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
To train all models:
```bash
python training.py
```

This will:
1. Preprocess the CICIDS 2017 dataset
2. Train the CNN model
3. Train the hierarchical SVM models
4. Train various Random Forest configurations

### Testing
To evaluate all models:
```bash
python testing.py
```

This will generate performance metrics and confusion matrices for all models.

## Dataset
This project uses the CICIDS 2017 dataset, which contains benign and the most up-to-date common attacks. The dataset reflects real-world network traffic and includes various attack scenarios.

## Requirements
See `requirements.txt` for a complete list of dependencies.

## Performance
The models are evaluated using various metrics including:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

Detailed performance metrics are generated during the testing phase.

## License
[Add your chosen license here]

## Contributing
Feel free to open issues or submit pull requests for any improvements.

## Authors
[Your Name]

## Acknowledgments
- CICIDS 2017 Dataset creators
- [Add any other acknowledgments]
