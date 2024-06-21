import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix


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
    "sensor_id1_AngY",
    "sensor_id2_AngX",
    "sensor_id5_AngY",
    "sensor_id4_AccZ",
    "sensor_id4_AngX",
    "machine_nameKuka Robot_power"]

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
        fig.write_html(f"./plots/{title}.html")
        #fig.write_image(f"./plots/{title}.png")
        print(f"Plot saved in ./plots/{title}.html and ../plots/{title}.png")