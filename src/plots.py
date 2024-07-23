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
    signals = df.columns
    
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
        
def plot_anomalies(anomaly_scores, freq, threshold):
    
    # Visualize the results
    plt.figure(figsize=(12, 6))

    scatter = plt.scatter(range(len(anomaly_scores)), anomaly_scores,
                        c=anomaly_scores, cmap='coolwarm',
                        norm=plt.Normalize(vmin=0, vmax=threshold*2))

    plt.colorbar(scatter, label='Anomaly Score')
    plt.axhline(y=threshold, color='g', linestyle='--', label='Threshold')
    plt.title(f'Anomaly Scores from LSTMED at frequency {freq}')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.show()

    anomalies_detected = sum(anomaly_scores > threshold)
    return anomalies_detected

def plot_anomalies_true_and_predicted(df, df_action, collisions_zones, df_predicted_zones, title="Some signals", saveplot=False):
    fig = go.Figure()
    signals = []

    collisions_zones = convert_to_df(collisions_zones)
    
    start = df.index[0]
    df_reduced = df.loc[start:]
    duration = 3600 * 3  # seconds
    time_delta = df_reduced.index - start
    df_interval = df_reduced[time_delta.total_seconds() <= duration]

    # Plot signals
    if signals:
        df_signals = df_interval[signals].select_dtypes(['number'])
        df_signals = df_signals / df_signals.max()
        for col in df_signals.columns:
            fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals[col], mode='lines', name=col))

    # Plot actions
    colors_action = px.colors.qualitative.Antique
    for j, action in enumerate(df_action.loc[df_interval.index].action.unique()):
        df_action_interval = df_action.loc[df_interval.index]
        df_action_single_action = df_action_interval[df_action_interval['action'] == action]
        fig.add_trace(go.Scatter(
            x=df_action_single_action.index,
            y=[0] * len(df_action_single_action.index),
            line_shape="hv",
            line=dict(color=colors_action[j % len(colors_action)], width=2.5),
            name=action))

    # Highlight labeled collision zones at the bottom
    for _, row in collisions_zones.iterrows():
        fig.add_shape(
            type="rect",
            x0=row['start'], x1=row['end'],
            y0=-1.0, y1=-0.0,
            fillcolor="blue", opacity=0.5,
            layer="below", line_width=0,
        )

    # Highlight predicted collision zones at the top
    for _, row in df_predicted_zones.iterrows():
        if row['is_collision'] == 1:
            fig.add_shape(
                type="rect",
                x0=row['start'], x1=row['end'],
                y0=0.0, y1=1.0,
                fillcolor="red", opacity=0.5,
                layer="below", line_width=0,
            )

    # Add legend for labeled and predicted anomalies
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='blue'),
        legendgroup="labeled", showlegend=True, name="Labeled Anomalies"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='red'),
        legendgroup="predicted", showlegend=True, name="Predicted Anomalies"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="",
        legend_title="Legend",
        font=dict(family="Courier New, monospace", size=12, color="Black"),
        yaxis=dict(range=[-1, 1]),
        showlegend=True
    )

    fig.show()
    
    if saveplot:
        fig.write_html(f"../plots/{title}.html")
        print(f"Plot saved in ../plots/{title}.html")
    
def plot_roc_curve(true_labels, anomaly_scores):
    # Ensure y_true is a numpy array if it's a DataFrame column

    fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
        
def convert_to_df(collisions_zones):
    collisions_zones_df = pd.DataFrame(collisions_zones)
    # change the type of the columns to datetime
    collisions_zones_df['start'] = pd.to_datetime(collisions_zones_df['start'])
    collisions_zones_df['end'] = pd.to_datetime(collisions_zones_df['end'])
    
    return collisions_zones_df