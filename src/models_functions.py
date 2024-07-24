import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_recall_curve

from dataset import *
from plots import *
from metrics import *

def get_dataframes(root, file_name, recording, freq, features_folder=None):
    # print the arguments
    fp_csv = [os.path.join(root, f"rec{r}{file_name}{freq}s.csv") for r in recording]
    fp_meta = [os.path.join(root, f"rec{r}{file_name}{freq}s.metadata") for r in recording]
    df_action, df, df_meta, action2int = get_df_action(fp_csv, fp_meta)
    frequency = 1/float(freq)
    start_time = time.time()
    df_features = get_features_ts("statistical", df_action, df_meta, frequency, action2int, features_folder)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return df_features, df, df_action

def get_collisions(recording, root):
    xls = pd.ExcelFile(os.path.join(root, "20220811_collisions_timestamp.xlsx"))
    collisions_rec = pd.read_excel(xls, f"rec{recording}")
    collisions_adjusted_rec = collisions_rec.Timestamp - pd.to_timedelta([2] * len(collisions_rec.Timestamp), unit='h')
    collisions_rec["Timestamp"] = collisions_adjusted_rec
    collisions_init = collisions_rec[collisions_rec['Inizio/fine'] == 'i'].Timestamp
    
    return collisions_rec, collisions_init

def get_collisions_zones_and_labels(collisions_rec, collisions_init, df_features):
    collisions_zones = find_collisions_zones(collisions_rec)
    df_with_labels = label_collision_data(df_features, collisions_init)
    y_collisions = df_with_labels["is_collision"]
    df_features.drop(columns=["is_collision"], inplace=True)
    
    return collisions_zones, y_collisions

def plot_precision_recall_curve(y_true, y_scores):
    # Compute precision-recall pairs for different probability thresholds
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    # Compute the area under the curve (AUC) using the trapezoidal rule
    auc_score = auc(recall, precision)
    
    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'Precision-Recall curve (AUC: {auc_score:.2f})')
    
    # Add labels and legend
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    
    # Show grid
    plt.grid(True)
    
    plt.show()
    
def find_best_threshold(y_true, scores):
    # Generate precision, recall, and corresponding thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    # Remove NaN values that may arise from division by zero in F1 calculation
    f1_scores = np.nan_to_num(f1_scores)
    
    # Find index of the best F1 score
    best_idx = np.argmax(f1_scores)
    
    # Best values
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0  # Handle edge case where the best index is out of threshold array bounds
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    
    return best_threshold, best_f1, best_precision, best_recall

def get_statistics(X_test, y_collisions_true, classifier, df_test, freq, threshold_type="std"):
    anomaly_score = predict_anomaly_score(X_test, classifier)
    thresholds = compute_various_thresholds(anomaly_score)
    
    if threshold_type == "std":
        threshold = thresholds[0]
    elif threshold_type == "mad":
        threshold = thresholds[1]
    elif threshold_type == "percentile":
        threshold = thresholds[2]
    elif threshold_type == "IQR":
        threshold = thresholds[3]
    elif threshold_type == "zero":
        threshold = thresholds[4]
    else:
        print("Invalid threshold type. Choose one of the following: std, mad, percentile, IQR")
    
    y_collisions_predict = anomaly_score >= threshold
    df_test["anomaly_score"] = anomaly_score
    df_test["is_collision"] = y_collisions_predict
    print(f"choosen threshold type: {threshold_type}, with value: {threshold:.4f}")
    compute_metrics(y_collisions_true, y_collisions_predict)
    roc_auc = roc_auc_score(y_collisions_true, anomaly_score)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    plot_roc_curve(y_collisions_true, anomaly_score)
    anomalies_detected = plot_anomalies(anomaly_score, freq, threshold)
    plot_precision_recall_curve(y_collisions_true, anomaly_score)
    print(f"Anomalies detected: {anomalies_detected}")
    
    best_threshold, best_f1, best_precision, best_recall = find_best_threshold(y_collisions_true, anomaly_score)
    print(f"Best threshold: {best_threshold:.4f} | F1 Score: {best_f1:.4f} | Precision: {best_precision:.4f} | Recall: {best_recall:.4f}")
    y_collisions_predict_best = sum(anomaly_score >= best_threshold)
    print(f"Anomalies detected with best threshold: {y_collisions_predict_best}")
    print(f"\n\t-------------------------------------------------------------------------------------\n")
    return df_test