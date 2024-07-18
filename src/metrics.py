import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, classification_report

class Confidence:

    def __init__(self, multiple_preds):
        self.multiple_preds = multiple_preds
        self._uncertainty_metrics = dict()
        self._uncertainty_metrics['entropy'] = self._entropy
        self._uncertainty_metrics['variance'] = self._variance
        self._uncertainty_metrics['max_softmax_response'] = self._max_softmax_response

    def compute_uncertainty_metrics(self):
        return {metric: self._compute_uncertainty(
            metric,
            self.multiple_preds) for metric in self._uncertainty_metrics.keys()}

    def _normalize(self, values):
            return (values - values.min())/(values.max()-values.min())

    def _compute_uncertainty(self, metric, multiple_preds):
        try:
            print("Done {}".format(metric))
            return self._normalize(
                self._uncertainty_metrics[metric](multiple_preds))
        except KeyError:
            print("{} not implemented.".format(metric))

    def _avreage_prediction(self, multiple_preds):
        if len(multiple_preds.shape) > 2:
            return np.mean(np.array(multiple_preds), axis=0)
        else:
            return multiple_preds

    def _entropy(self, multiple_preds):
        avg_preds = self._avreage_prediction(multiple_preds)
        eps = 1e-5
        entropy = -1 * np.sum(avg_preds * np.log(avg_preds + eps), axis=1)
        return entropy

    def _variance(self, multiple_preds):
        avg_preds = self._avreage_prediction(multiple_preds)
        return  np.var(avg_preds, axis=1)

    def _max_softmax_response(self, multiple_preds):
        avg_preds = self._avreage_prediction(multiple_preds)
        return np.max(avg_preds, axis=1)

def anomaly_detection_metric(anomaly_start_timestamps, confidence, df_dataset, thresholds, less_than=True):
    "Actual is y axis"
    if not less_than:
        confidence = 1 - confidence

    sens = list()
    spec = list()
    fpr = list()
    f1 = list()
    prec = list()
    cm_list = list()
    anomaly_indexes_dict = dict()
    acc_with_err = list()
    for threshold in thresholds:
        df_not_confident = df_dataset[confidence <= threshold]
        tp = 0
        anomaly_indexes = list()
        for anomaly in anomaly_start_timestamps:
            for index, row in df_not_confident.iterrows():
                if anomaly >= row['start'] and anomaly <= row['end']:
                    anomaly_indexes.append(index)
                    tp += 1

        cm_anomaly = np.zeros((2, 2))
        n_samples = len(df_dataset)
        n_not_collisions = n_samples - len(anomaly_start_timestamps)
        n_detected = len(df_not_confident)

        fp = n_detected - tp
        fn = len(anomaly_start_timestamps) - tp
        tn = n_not_collisions - fp

        cm_anomaly[0][0] = tn
        cm_anomaly[1][1] = tp
        cm_anomaly[0][1] = fp
        cm_anomaly[1][0] = fn
        cm_list.append(cm_anomaly)
        sens.append(tp / (tp + fn))
        recall = tp / (tp + fn)
        prec.append(tp / (tp + fp))
        spec.append(tn / (fp + tn))
        fpr.append(1 - tn / (fp + tn))
        try:
            f1.append(2 * tp / (2 * tp + fp + fn) )
        except ZeroDivisionError:
            f1.append(0)
        cm_anomaly_norm = cm_anomaly.astype('float') / cm_anomaly.sum(axis=1)[:, np.newaxis]
        acc_with_err.append((np.mean(np.diag(cm_anomaly_norm)),
                            np.std(np.diag(cm_anomaly_norm))))


        anomaly_indexes_dict[threshold] = anomaly_indexes
    return sens, spec, fpr, f1, cm_list, anomaly_indexes_dict, acc_with_err, prec

def compute_metrics(anomaly_scores, y_test, threshold):
        
    # Find the class with the highest average anomaly score
    class_avg_scores = {}
    for class_label in np.unique(y_test):
        class_avg_scores[class_label] = np.mean(anomaly_scores[y_test == class_label])

    anomaly_class = max(class_avg_scores, key=class_avg_scores.get)

    # Create binary labels: 1 for the anomaly class, 0 for others
    y_test_binary = (y_test == anomaly_class).astype(int)

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test_binary, anomaly_scores)

    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Detected anomaly class: {anomaly_class}")
    
    print(f"Threshold: {threshold:.4f}")
    
    # Calculate F1 score
    f1 = f1_score(y_test_binary, anomaly_scores > threshold)
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_binary, anomaly_scores > threshold)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Calculate precision
    precision = precision_score(y_test_binary, anomaly_scores > threshold)
    print(f"Precision: {precision:.4f}")
    
    # Calculate recall
    recall = recall_score(y_test_binary, anomaly_scores > threshold)
    print(f"Recall: {recall:.4f}")
    
    print(classification_report(y_test_binary, anomaly_scores > threshold))
    
def compute_various_thresholds(anomaly_scores):
    # Compute the mean and standard deviation of the anomaly scores
    threshold_1 = np.mean(anomaly_scores) + 2 * np.std(anomaly_scores)
    # Compute the median and median absolute deviation of the anomaly scores
    median = np.median(anomaly_scores)
    mad = np.median(np.abs(anomaly_scores - median))
    threshold_2 = median + 2 * mad  
    # Compute the 95th percentile of the anomaly scores
    threshold_3 = np.percentile(anomaly_scores, 95)
    # Compute the interquartile range of the anomaly scores
    Q1 = np.percentile(anomaly_scores, 25)
    Q3 = np.percentile(anomaly_scores, 75)
    IQR = Q3 - Q1
    threshold_4 = Q3 + 1.5 * IQR
    ["std", "mad", "percentile", "IQR"]
    
    for threshold, name in zip ([threshold_1, threshold_2, threshold_3, threshold_4], ["std", "mad", "percentile", "IQR"]):
        anomalies_detected = sum(anomaly_scores >= threshold)
        print(f"Number of anomalies detected: {anomalies_detected} with threshold {threshold}, {name}")
    print()
    return threshold_1, threshold_2, threshold_3, threshold_4