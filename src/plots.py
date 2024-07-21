import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import matplotlib.dates as mdates
from sklearn.metrics import confusion_matrix, roc_curve, auc

def seaborn_cm(cm, ax, tick_labels, fontsize=14, title=None, sum_actual="over_columns",
               xrotation=0, yrotation=0):
    """
    Function to plot a confusion matrix
    """
    from matplotlib import cm as plt_cmap
    group_counts = ["{:0.0f}".format(value) for value in cm.flatten()]
    if sum_actual == "over_columns":
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif sum_actual == "over_rows":
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    else:
        print("sum_actual must be over_columns or over_rows")
        exit()
    cm = np.nan_to_num(cm)
    mean_acc = np.mean(np.diag(cm)[cm.sum(axis=1) != 0])
    std_acc = np.std(np.diag(cm))
    group_percentages = ["{:0.0f}".format(value*100) for value in cm.flatten()]
    cm_labels = [f"{c}\n{p}%" for c, p in zip(group_counts, group_percentages)]
    cm_labels = np.asarray(cm_labels).reshape(len(tick_labels), len(tick_labels))
    sns.heatmap(cm,
                ax=ax,
                annot=cm_labels,
                fmt='',
                cbar=False,
                cmap="Blues",
                linewidths=1, linecolor='black',
                annot_kws={"fontsize": fontsize},
                xticklabels=tick_labels,
                yticklabels=tick_labels)
    ax.set_yticklabels(ax.get_yticklabels(), size=fontsize, rotation=yrotation)
    ax.set_xticklabels(ax.get_xticklabels(), size=fontsize, rotation=xrotation)
    if title:
        title = f"{title}\nMean accuracy {mean_acc * 100:.1f} +- {std_acc * 100:.1f}"
    else:
        title = f"Mean accuracy {mean_acc * 100:.1f} +- {std_acc * 100:.1f}"
    ax.set_title(title)
    if sum_actual == "over_columns":
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    else:
        ax.set_ylabel("Predicted")
        ax.set_xlabel("Actual")
    ax.axis("off")

def create_and_plot_cm(y_pred, y_true, action2int):
    cm = confusion_matrix(y_true, y_pred.argmax(axis=1), labels=list(action2int.values()))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    int2action = {v: k for k, v in action2int.items()}
    seaborn_cm(cm,
                ax,
                [int2action[l] for l in action2int.values()], fontsize=8, xrotation=90)
    plt.tight_layout()
    
def plot_uncertainty(uncertainties, title):
    fig, axes = plt.subplots(len(uncertainties['correct'].keys()), 3, figsize=(15, 9))
    for ax, measure in zip(axes, uncertainties['correct'].keys()):
        ax[0].set_title(f"Wrong - {measure}")
        ax[0].hist(uncertainties['wrong'][measure], color="red", log=False, bins=25, edgecolor='black', linewidth=1.2, alpha=0.5);
        ax[1].set_title(f"Correct - {measure}")
        ax[1].hist(uncertainties['correct'][measure], color="green", log=False, bins=25, edgecolor='black', linewidth=1.2, alpha=0.5);
        ax[2].set_title(f"All - {measure}")
        ax[2].hist(uncertainties['all'][measure], color="blue", log=False, bins=25, edgecolor='black', linewidth=1.2, alpha=0.5);
    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()
    
def plot_signals(df, df_action, title="Some signals", saveplot=False):
    fig = go.Figure()
    signals = [
    "machine_nameKuka Robot_apparent_power",
    "machine_nameKuka Robot_current",
    "machine_nameKuka Robot_export_reactive_energy",
    "machine_nameKuka Robot_frequency",
    "machine_nameKuka Robot_import_active_energy",
    "machine_nameKuka Robot_phase_angle",
    "machine_nameKuka Robot_power",
    "machine_nameKuka Robot_power_factor",
    "machine_nameKuka Robot_reactive_power",
    "machine_nameKuka Robot_voltage",
    "sensor_id1_AccX",
    "sensor_id1_AccY",
    "sensor_id1_AccZ",
    "sensor_id1_AngX",
    "sensor_id1_AngY",
    "sensor_id1_AngZ",
    "sensor_id1_GyroX",
    "sensor_id1_GyroY",
    "sensor_id1_GyroZ",
    "sensor_id2_AccX",
    "sensor_id2_AccY",
    "sensor_id2_AccZ",
    "sensor_id2_AngX",
    "sensor_id2_AngY",
    "sensor_id2_AngZ",
    "sensor_id2_GyroX",
    "sensor_id2_GyroY",
    "sensor_id2_GyroZ",
    "sensor_id3_AccX",
    "sensor_id3_AccY",
    "sensor_id3_AccZ",
    "sensor_id3_AngX",
    "sensor_id3_AngY",
    "sensor_id3_AngZ",
    "sensor_id3_GyroX",
    "sensor_id3_GyroY",
    "sensor_id3_GyroZ",
    "sensor_id4_AccX",
    "sensor_id4_AccY",
    "sensor_id4_AccZ",
    "sensor_id4_AngX",
    "sensor_id4_AngY",
    "sensor_id4_AngZ",
    "sensor_id4_GyroX",
    "sensor_id4_GyroY",
    "sensor_id4_GyroZ",
    "sensor_id5_AccX",
    "sensor_id5_AccY",
    "sensor_id5_AccZ",
    "sensor_id5_AngX",
    "sensor_id5_AngY",
    "sensor_id5_AngZ",
    "sensor_id5_GyroX",
    "sensor_id5_GyroY",
    "sensor_id5_GyroZ",]
    
    start = df.index[9000]
    df_reduced = df.loc[start:]
    duration = 3600 * 3  # seconds
    time_delta = df_reduced.index - start
    df_interval = df_reduced[time_delta.total_seconds() <= duration]
    j = 0

    # Leveraging plotly express
    n_colors = len(signals)
    colors = px.colors.sample_colorscale("greys", [n/(n_colors -1) for n in range(n_colors)])  # From continuous colormap
    colors = px.colors.qualitative.Set2  # From discrete colormap, see https://plotly.com/python/discrete-color/
    df_signals = df_interval[signals].select_dtypes(['number'])
    df_signals = df_signals / df_signals.max()
    fig = px.line(df_signals, x=df_signals.index, y=df_signals.columns, color_discrete_sequence=colors)

    # Leveraging plotly graph object
    colors_action = px.colors.qualitative.Antique
    j = 0
    for action in df_action.loc[df_interval.index].action.unique():
        df_action_interval = df_action.loc[df_interval.index]
        df_action_single_action = df_action_interval[df_action_interval['action'] == action]
        fig.add_trace(go.Scatter(
            x=df_action_single_action.index,
            y=[-0.3] * len(df_action_single_action.index),
            line_shape="hv",
            line=dict(color=colors_action[j % len(colors_action)], width=2.5),
            name=action))
        j += 1


    fig.update_layout(
    title=title,
    xaxis_title="Time",
    yaxis_title="",
    legend_title="Legend",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="Black"
    )
    )
    fig.show()
    
    if saveplot:
        fig.write_html(f"../plots/{title}.html")
        #fig.write_image(f"../plots/{title}.png")
        print(f"Plot saved in ../plots/{title}.html and ../plots/{title}.png")
        
def plot_anomalies(anomaly_scores, freq, threshold, model_name="No_Model"):
    
    # Visualize the results
    plt.figure(figsize=(12, 6))

    scatter = plt.scatter(range(len(anomaly_scores)), anomaly_scores,
                        c=anomaly_scores, cmap='coolwarm',
                        norm=plt.Normalize(vmin=0, vmax=threshold*2))

    plt.colorbar(scatter, label='Anomaly Score')
    plt.axhline(y=threshold, color='g', linestyle='--', label='Threshold')
    plt.title(f'Anomaly Scores from {model_name} at frequency {freq}')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.show()

    anomalies_detected = sum(anomaly_scores > threshold)
    print(f"Number of anomalies detected: {anomalies_detected}")
    
    return anomalies_detected

def plot_all_anomalies_over_time(X_test, anomaly_scores, anomalies_detected, freq):
    # Step 1: Create a DataFrame with the original data and anomaly scores
    df = pd.DataFrame(X_test)
    df['anomaly_score'] = pd.Series(anomaly_scores)

    # Step 2: Add a timestamp column since it doesn't exist
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='T')

    # Step 3: Select a few features to plot
    features_to_plot = df.columns
    features_to_plot = features_to_plot.drop(['anomaly_score', 'timestamp'])

    # Step 4: Create the plot
    fig, axs = plt.subplots(len(features_to_plot) + 1, 1, figsize=(15, 5*len(features_to_plot)), sharex=True)
    fig.suptitle(f'Time Series Data with Anomaly Scores at frequency {freq}', fontsize=16)

    for i, feature in enumerate(features_to_plot):
        axs[i].plot(df['timestamp'], df[feature], label=feature)
        axs[i].set_ylabel(feature)
        axs[i].legend(loc='upper left')

    # Plot anomaly scores
    axs[-1].plot(df['timestamp'], df['anomaly_score'], color='red', label='Anomaly Score')
    axs[-1].set_ylabel('Anomaly Score')
    axs[-1].set_xlabel('Time')
    axs[-1].legend(loc='upper left')

     # Highlight top N anomalies
    N = anomalies_detected
    top_anomalies = df.nlargest(N, 'anomaly_score')

    for ax in axs:
        for idx, row in top_anomalies.iterrows():
            ax.axvline(x=row['timestamp'], color='green', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

     # Print details of top anomalies
    print("Top", N, "Anomalies:")
    print(top_anomalies[['timestamp', 'anomaly_score'] + list(features_to_plot)])

def plot_anomalies_over_time(X_test, anomaly_scores, anomalies_detected, freq, threshold, collision_zones, X_test_start_end):
    # Step 1: Create a DataFrame with the original data and anomaly scores
    df = pd.DataFrame(X_test)
    df['anomaly_score'] = pd.Series(anomaly_scores)

    # Step 2: Add a timestamp column since it doesn't exist
    df['timestamp'] = pd.date_range(start=X_test_start_end['start'].to_list()[0], end=X_test_start_end['end'].to_list()[-1], periods=len(df))  #Qua assicurati di bindare bene i timestamp (se puoi, non generarli)

    # Step 3: Select a few features to plot along with the anomaly scores
    features_to_plot = df.columns.drop(['anomaly_score', 'timestamp'])

    # Step 4: Create the plot
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.set_title(f'Time Series Data with Anomaly Scores at frequency {freq}', fontsize=16)
    
    # Plot features on primary y-axis
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Feature Values')
    colors = plt.cm.Greys(np.linspace(0, 1, len(features_to_plot)))
    lines = []  # To collect plot lines for legend
    labels = []  # To collect plot labels for legend
    
    # print(f"THRESH: {threshold}")
    for feature, color in zip(features_to_plot, colors):
        # if any(df[feature] > threshold):
        line, = ax1.plot(df['timestamp'], df[feature], label=f'Feature: {feature}', linewidth=1, color=color, alpha=0.7)
        lines.append(line)
        labels.append(f'Feature: {feature}')
        # else:
            # print("\t\tQuesta feature non supera mai soglia")

    #Highlighting collision zones on the graph
    for s, e in zip(collision_zones['start'].tolist(), collision_zones['end'].tolist()):
        ax1.axvspan(s, e, alpha=0.2, color='blue')

    # Highlight top N anomalies
    N = anomalies_detected
    top_anomalies = df.nlargest(N, 'anomaly_score')
    for time in top_anomalies['timestamp']:
        ax1.axvline(x=time, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Detected Anomaly (Top N)')

    # ax1.axhline(y=threshold, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Threshold')

    #dATO CHE AX1 E AX2 SONO SULLA STESSA SCALA HO TOLTO AX2
    # Plot anomaly scores on secondary y-axis
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Anomaly Score', color='red')
    # ax2.plot(df['timestamp'], df['anomaly_score'], color='red', label='Anomaly Score', linestyle='--', linewidth=1)
    # ax2.tick_params(axis='y', labelcolor='red')

    ax1.plot(df['timestamp'], df['anomaly_score'], color='red', label='Anomaly Score', linestyle='--', linewidth=1)

    fig.tight_layout()
    plt.show()

    # Print details of top anomalies
    print("Top", N, "Anomalies:")
    print(top_anomalies[['timestamp', 'anomaly_score'] + list(features_to_plot)])

def plot_anomalies_over_time_isolation_forest(X_test, pred, freq, collision_zones, X_test_start_end):
    # Step 1: Create a DataFrame with the original data and anomaly scores
    df = X_test.copy()

    # Step 2: Add a timestamp column since it doesn't exist
    df['timestamp'] = pd.date_range(start=X_test_start_end['start'].to_list()[0], end=X_test_start_end['end'].to_list()[-1], periods=len(df))  #Qua assicurati di bindare bene i timestamp (se puoi, non generarli)

    # Step 3: Select a few features to plot along with the anomaly scores
    features_to_plot = df.columns.drop(['timestamp'])

    # Step 4: Create the plot
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.set_title(f'Time Series Data with Anomaly Scores at frequency {freq}', fontsize=16)
    
    # Plot features on primary y-axis
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Feature Values')
    colors = plt.cm.Greys(np.linspace(0, 1, len(features_to_plot)))
    lines = []  # To collect plot lines for legend
    labels = []  # To collect plot labels for legend
    
    # print(f"THRESH: {threshold}")
    for feature, color in zip(features_to_plot, colors):
        # if any(df[feature] > threshold):
        line, = ax1.plot(df['timestamp'], df[feature], label=f'Feature: {feature}', linewidth=1, color=color, alpha=1)
        lines.append(line)
        labels.append(f'Feature: {feature}')
        # else:
            # print("\t\tQuesta feature non supera mai soglia")

    #Highlighting collision zones on the graph
    for s, e in zip(collision_zones['start'].tolist(), collision_zones['end'].tolist()):
        ax1.axvspan(s, e, alpha=0.2, color='blue')

    # ax1.plot(df['timestamp'], df['anomaly_score'], color='red', label='Anomaly Score', linestyle='--', linewidth=1)
    ax1.plot(df['timestamp'], pred, 'go-', label='Anomaly Score', linestyle='--', linewidth=1)

    fig.tight_layout()
    plt.show()
    
def plot_roc_curve(y_true, anomaly_scores):
    # Ensure y_true is a numpy array if it's a DataFrame column

    # Calculate ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    roc_auc = auc(fpr, tpr)

    # Plotting
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()