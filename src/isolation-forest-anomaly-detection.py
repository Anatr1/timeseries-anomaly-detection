import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import time
from dataset import get_df_action, get_features_ts, get_train_test_data
from plots import plot_anomalies, plot_anomalies_over_time, plot_roc_curve
from scipy.stats import mode


# Set style for matplotlib
plt.style.use("Solarize_Light2")

# Path to the root directory of the dataset
ROOTDIR_DATASET_NORMAL = '../../dataset/normal'
ROOTDIR_DATASET_ANOMALY = '../../dataset/collisions'

# Disable TensorFlow OneDNN optimization
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Data Loading and Preprocessing
freq = '0.1'

# Load normal data
filepath_csv = [os.path.join(ROOTDIR_DATASET_NORMAL, f"rec{r}_20220811_rbtc_{freq}s.csv") for r in [0, 2, 3, 4]]
filepath_meta = [os.path.join(ROOTDIR_DATASET_NORMAL, f"rec{r}_20220811_rbtc_{freq}s.metadata") for r in [0, 2, 3, 4]]
df_action, df, df_meta, action2int = get_df_action(filepath_csv, filepath_meta)

# Load collision data
filepath_csv = [os.path.join(ROOTDIR_DATASET_ANOMALY, f"rec{r}_collision_20220811_rbtc_{freq}s.csv") for r in [1, 5]]
filepath_meta = [os.path.join(ROOTDIR_DATASET_ANOMALY, f"rec{r}_collision_20220811_rbtc_{freq}s.metadata") for r in [1, 5]]
df_action_collision, df_collision, df_meta_collision, action2int_collision = get_df_action(filepath_csv, filepath_meta)

# Feature Extraction
start_time = time.time()
frequency = 1/float(freq)
df_features = get_features_ts("statistical", df_action, df_meta, frequency, action2int)
df_features_collision = get_features_ts("statistical", df_action_collision, df_meta_collision, frequency, action2int_collision)
print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")

# Prepare train and test data
X_train, y_train, X_test, y_test = get_train_test_data(df_features, df_features_collision, full_normal=True)

# Convert to numpy arrays if they're not already
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create sliding windows
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10  # Adjust this based on your data's characteristics
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps)

# Reshape the data for Isolation Forest
X_train_reshaped = X_train_seq.reshape(X_train_seq.shape[0], -1)
X_test_reshaped = X_test_seq.reshape(X_test_seq.shape[0], -1)

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train_reshaped)

plot_anomalies(iso_forest, X_test_reshaped, y_test, freq)

# Predict anomalies
anomaly_scores = -iso_forest.score_samples(X_test_reshaped)

# Plot anomalies
plt.figure(figsize=(12, 6))
plt.plot(anomaly_scores, label='Anomaly Score')
plt.axhline(y=np.percentile(anomaly_scores, 90), color='r', linestyle='--', label='Threshold')
plt.title('Anomaly Scores using Isolation Forest')
plt.xlabel('Sample Index')
plt.ylabel('Anomaly Score')
plt.legend()
plt.show()

# Convert multiclass labels to binary (normal vs anomaly)
normal_class = mode(y_train_seq)[0][0]  # Assume the most frequent class is normal
y_test_binary = (y_test_seq != normal_class).astype(int)

# Compute metrics
threshold = np.percentile(anomaly_scores, 90)
y_pred = (anomaly_scores > threshold).astype(int)

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test_binary, anomaly_scores)
average_precision = average_precision_score(y_test_binary, anomaly_scores)

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Map anomalies to original time series
anomalies_detected = sum(y_pred)
X_test_seq_flattened = X_test_seq.reshape(X_test_seq.shape[0], -1)
plot_anomalies_over_time(X_test_seq_flattened, anomaly_scores, anomalies_detected, freq)

# Print some statistics
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of X_train_seq: {X_train_seq.shape}")
print(f"Shape of X_test_seq: {X_test_seq.shape}")
print(f"Number of anomalies detected: {anomalies_detected}")
print(f"Percentage of anomalies: {anomalies_detected / len(y_pred) * 100:.2f}%")
print(f"Normal class: {normal_class}")
print(f"Unique classes in test set: {np.unique(y_test_seq)}")
print(f"Average Precision Score: {average_precision:.2f}")
