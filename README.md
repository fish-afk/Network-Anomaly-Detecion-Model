
# Network Anomaly Detector

This project implements anomaly detection on network traffic using a Random Forest classifier and the NSL-KDD dataset. The goal is to identify data points that deviate significantly from normal network behavior — these anomalies may represent malicious activities, intrusions, or other security breaches.

## Anomaly Detection

Anomaly detection identifies patterns in data that do not conform to expected behavior. In cybersecurity, detecting anomalies is crucial for:

- Spotting network intrusions
- Identifying suspicious or malicious activities
- Strengthening defense mechanisms

## Random Forests for Anomaly Detection

A Random Forest is an ensemble machine learning algorithm that builds multiple decision trees and combines their outputs.

Key concepts:
- Bootstrapping: Each tree is trained on a random sample (with replacement) of the dataset.
- Random feature selection: Each split in a tree considers a random subset of features to promote diversity.
- Voting/Averaging: Final prediction is based on majority voting (classification) or averaging (regression).

When applied to anomaly detection:
- The model learns normal network behavior.
- New data points that don’t fit this behavior are flagged as potential anomalies.

## Dataset: NSL-KDD

We use the NSL-KDD dataset, an improved version of the KDD Cup 1999 dataset. It:
- Removes redundant records
- Balances class distributions
- Supports both binary (normal vs. attack) and multi-class classification

Our project uses a modified version of this dataset (`KDD+.txt`).

## Downloading and Loading the Dataset

### Download

```python
import requests, zipfile, io

url = "https://academy.hackthebox.com/storage/modules/292/KDD_dataset.zip"

# Download and extract
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall('.')  # Extract to current directory
```

### Load into DataFrame

```python
import numpy as np
import pandas as pd

file_path = r'KDD+.txt'

# Define column names
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
]

# Load dataset
df = pd.read_csv(file_path, names=columns)

# View first rows
print(df.head())
```

## Libraries Used

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
```

- numpy, pandas -> data manipulation
- RandomForestClassifier -> anomaly detection model
- scikit-learn metrics -> model evaluation
- seaborn, matplotlib -> data and result visualization

## Model Overview

- The dataset is split into training and testing sets.
- A Random Forest classifier is trained on this data.
- The model predicts whether a network connection is normal or an attack.
- Performance is evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
  - Classification report

## Running the Notebook

Open `network_anomalty_detector.ipynb` and execute the cells in sequence. The notebook:
- Loads and processes the data
- Trains the model
- Evaluates performance
- Visualizes results

## Future Enhancements

- Add hyperparameter tuning for the Random Forest
- Explore feature importance and selection
- Try alternative models (e.g., Isolation Forest, Autoencoders)
- Deploy as a real-time detection service
