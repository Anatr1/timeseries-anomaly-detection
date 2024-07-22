import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tsfel
import pandas as pd
import warnings
import os
from pathlib import Path
import numpy as np
from pandas import HDFStore

from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold

def get_train_test_split(data_path, data_type='npz', test_size=0.2, shuffle=False):
    data = np.load(data_path)
    if data_type == 'npz':
        data = data[data.files[0]]
    labels = data[:, -1]
    features = data[:, :-1]
    return train_test_split(features, labels, test_size=test_size, shuffle=shuffle)

def get_df_action(filepaths_csv, filepaths_meta, action2int=None, delimiter=";"):
    # Load dataframes
    print("Loading data.")
    # Make dataframes
    # Some classes show the output boolean parameter as True rather than true. Fix here
    dfs_meta = list()
    for filepath in filepaths_meta:
        df_m = pd.read_csv(filepath, sep=delimiter)
        df_m.str_repr = df_m.str_repr.str.replace('True', 'true')
        df_m['filepath'] = filepath
        dfs_meta.append(df_m)

    df_meta = pd.concat(dfs_meta)
    df_meta.index = pd.to_datetime(df_meta.init_timestamp.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    df_meta['completed_timestamp'] = pd.to_datetime(df_meta.completed_timestamp.astype('datetime64[ms]'),
                                                    format="%Y-%m-%dT%H:%M:%S.%f")
    df_meta['init_timestamp'] = pd.to_datetime(df_meta.init_timestamp.astype('datetime64[ms]'),
                                               format="%Y-%m-%dT%H:%M:%S.%f")

    # Eventually reduce number of classes
    # df_meta['str_repr'] = df_meta.str_repr.str.split('=', expand = True,n=1)[0]
    # df_meta['str_repr'] = df_meta.str_repr.str.split('(', expand=True, n=1)[0]

    actions = df_meta.str_repr.unique()
    dfs = [pd.read_csv(filepath_csv, sep=";") for filepath_csv in filepaths_csv]
    df = pd.concat(dfs)

    # Sort columns by name !!!
    df = df.sort_index(axis=1)

    # Set timestamp as index
    df.index = pd.to_datetime(df.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    # Drop useless columns
    columns_to_drop = [column for column in df.columns if "Abb" in column or "Temperature" in column]
    df.drop(["machine_nameKuka Robot_export_active_energy",
             "machine_nameKuka Robot_import_reactive_energy"] + columns_to_drop, axis=1, inplace=True)
    signals = df.columns

    df_action = list()
    for action in actions:
        for index, row in df_meta[df_meta.str_repr == action].iterrows():
            start = row['init_timestamp']
            end = row['completed_timestamp']
            df_tmp = df.loc[start: end].copy()
            df_tmp['action'] = action
            # Duration as string (so is not considered a feature)
            df_tmp['duration'] = str((row['completed_timestamp'] - row['init_timestamp']).total_seconds())
            df_action.append(df_tmp)
    df_action = pd.concat(df_action, ignore_index=True)
    df_action.index = pd.to_datetime(df_action.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    df_action = df_action[~df_action.index.duplicated(keep='first')]

    # Drop NaN
    df = df.dropna(axis=0)
    df_action = df_action.dropna(axis=0)

    if action2int is None:
        action2int = dict()
        j = 1
        for label in df_action.action.unique():
            action2int[label] = j
            j += 1

    df_merged = df.merge(df_action[['action']], left_index=True, right_index=True, how="left")
    # print(f"df_merged len: {len(df_merged)}")
    # Where df_merged in NaN Kuka is in idle state
    df_idle = df_merged[df_merged['action'].isna()].copy()
    df_idle['action'] = 'idle'
    df_idle['duration'] = df_action.duration.values.astype(float).mean().astype(str)
    df_action = pd.concat([df_action, df_idle])

    # ile label must be 0 for debug mode
    action2int['idle'] = 0
    print(f"Found {len(set(df_action['action']))} different actions.")
    print("Loading data done.\n")

    return df_action, df, df_meta, action2int

def get_features_ts(domain, df_action, df_meta, frequency, action2int, save_dir='path/to/save'):
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, f"features_{domain}_{frequency}.csv")
    
    if os.path.exists(save_path):
        print("Loading features from file.")
        with pd.HDFStore(save_path) as store:
            dataframe_features = store['data']
        return dataframe_features
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    duration_dict = {1: 10, 10: 1, 100: 0.1, 200: 0.05}
    duration_min = duration_dict[int(frequency)]
    cfg = tsfel.get_features_by_domain(domain)
    dataframe_features = list()
    print("Computing features.")
    # Idle does not have associated timestamps. Window is set to 10 seconds
    df_by_action = df_action[df_action["action"] == "idle"].copy()
    X = tsfel.time_series_features_extractor(cfg,
                                             df_by_action.select_dtypes(['number']),
                                             fs=frequency,
                                             header_names=df_by_action.select_dtypes(['number']).columns + '-',
                                             window_size=int(frequency * 10),
                                             verbose=False)
    time = pd.to_datetime(df_by_action.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    X['start'] = [t[0] for t in
                  tsfel.utils.signal_processing.signal_window_splitter(time, window_size=int(frequency * 10))]
    X['duration'] = 10
    X['end'] = X['start'] + pd.to_timedelta(X['duration'], 's')
    X['label'] = action2int["idle"]
    X.drop('duration', inplace=True, axis=1)
    dataframe_features.append(X)
    actions = list(df_action.action.unique())
    actions.remove("idle")
    for action in actions:
        df_by_action = df_action[df_action["action"] == action].copy()
        df_meta_by_action = df_meta[df_meta['str_repr'] == action].copy()
        df_meta_by_action['start'] = pd.to_datetime(df_meta_by_action.init_timestamp.astype('datetime64[ms]'),
                                                    format="%Y-%m-%dT%H:%M:%S.%f")
        df_meta_by_action['end'] = pd.to_datetime(
            df_meta_by_action.completed_timestamp.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
        for _, row in df_meta_by_action.iterrows():
            df_by_action_by_event = df_by_action.loc[row["start"]: row["end"]]
            if len(df_by_action_by_event) < duration_min * frequency:
                print(f"Skipped feature extraction for {action} {row['start']} : {row['end']}.")
                continue

            X = tsfel.calc_window_features(cfg,
                                           df_by_action_by_event.select_dtypes(['number']),
                                           header_names=df_by_action_by_event.select_dtypes(['number']).columns + '-',
                                           fs=frequency,
                                           verbose=False)
            # print(X.shape)
            X['label'] = action2int[action]
            X['start'] = row['start']
            X['end'] = row['end']
            dataframe_features.append(X)

    dataframe_features = pd.concat(dataframe_features)
    with pd.HDFStore(save_path) as store:
        store['data'] = dataframe_features
    print(f"Features saved to {save_path}.")
    
    return dataframe_features

def load_data(PATH, frequency):
    # FIXME: dataset with collision has a collision timestamp, need to understand how to use it
    
    if "normal" in PATH:
        filepath_csv = [os.path.join(PATH, f"rec{r}_20220811_rbtc_{frequency}s.csv") for r in [0, 2, 3, 4]]
        filepath_metadata = [os.path.join(PATH, f"rec{r}_20220811_rbtc_{frequency}s.metadata") for r in [0, 2, 3, 4]]
    else:
        collisions = pd.read_excel(os.path.join(PATH, "20220811_collisions_timestamp.xlsx"))
        collisions_init = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta([2] * len(collisions[collisions['Inizio/fine'] == "i"].Timestamp), 'h')
        # FIXME: end then is not used ????
        filepath_csv = [os.path.join(PATH, f"rec{r}_collision_20220811_rbtc_{frequency}s.csv") for r in [1, 5]]
        filepath_metadata = [os.path.join(PATH, f"rec{r}_collision_20220811_rbtc_{frequency}s.metadata") for r in [1, 5]]
        
    # Load data
    df_action, df, df_meta, action2int = get_df_action(filepath_csv, filepath_metadata)
    
    return df_action, df, df_meta, action2int

def get_train_test_data_df(df_action, df_meta, action2int, return_df_not_split=False):
    start_time = time.time()
    df_features = get_features_ts("statistical", df_action, df_meta, 10, action2int)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    df_features.isnull().values.any()
    df_features_nonan_test = df_features.drop((df_features.columns[df_features.isna().any()].tolist()), axis=1)
    df_features_nonan_test.isnull().values.any(), df_features_nonan_test.shape
    
    df_features_nonan = df_features.dropna(axis=1)
    df_features_nonan.isnull().values.any(), df_features_nonan_test.shape
    
    df_train, df_test = train_test_split(df_features_nonan)
    
    X_train = df_train.drop(['label', 'start', 'end'], axis=1)
    y_train = df_train['label']
    X_test = df_test.drop(['label', 'start', 'end'], axis=1)
    y_test = df_test['label']
    
    if return_df_not_split:
        return df_train, df_test
    else:
        return X_train, y_train, X_test, y_test

def prepare_data_for_tf(X_train, y_train, X_test):
    # Normalise features
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

    # Remove zero-variance features
    selector_variance = VarianceThreshold()
    selector_variance.fit(X_train)
    X_train = pd.DataFrame(selector_variance.transform(X_train),
                            columns=X_train.columns.values[selector_variance.get_support()])

    # Remove highly correlated features
    corr_features = tsfel.correlated_features(X_train,
                                            threshold=0.95)
    X_train.drop(corr_features, inplace=True, axis=1)

    # Lasso selector
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
    lasso = SelectFromModel(lsvc, prefit=True)
    selected_features = X_train.columns.values[lasso.get_support()]
    X_train = X_train[selected_features].copy()

    # Labels
    num_classes = len(set(y_train))
    y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    # Test
    X_test = pd.DataFrame(selector_variance.transform(scaler.transform(X_test)),
                        columns=X_test.columns.values[selector_variance.get_support()])
    X_test.drop(corr_features, inplace=True, axis=1)
    X_test = X_test[selected_features].copy()
    
    return X_train, y_train_categorical, X_test

def get_train_test_data(df_features, df_features_collision, full_normal=True):
    df_features.isnull().values.any()
    #df_features_nonan = df_features.drop((df_features.columns[df_features.isna().any()].tolist()), axis=1)
    df_features_nonan = df_features.fillna(0)
    df_features_collision_nonan = df_features_collision.fillna(0)

    # I normally want to train on the whole normal dataset
    if not full_normal:
        df_train, df_test = train_test_split(df_features_nonan)
    else:
        df_train = df_features_nonan
        df_test = df_features_collision_nonan
        
    X_train = df_train.drop(["label", "start", "end"], axis=1)
    y_train = df_train["label"]
    X_test = df_test.drop(["label", "start", "end"], axis=1)
    y_test = df_test["label"]

    # Normalize features
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)

    # Remove zero-variance features
    selector_variance = VarianceThreshold()
    selector_variance.fit(X_train)
    X_train = pd.DataFrame(selector_variance.transform(X_train),
                            columns=X_train.columns.values[selector_variance.get_support()])

    # Remove highly correlated features
    corr_features = tsfel.correlated_features(X_train,
                                            threshold=0.95)
    X_train.drop(corr_features, inplace=True, axis=1)

    # Lasso selector
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
    lasso = SelectFromModel(lsvc, prefit=True)
    selected_features = X_train.columns.values[lasso.get_support()]
    X_train = X_train[selected_features].copy()

    # Labels
    num_classes = len(set(y_train))
    y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    # Test
    X_test = pd.DataFrame(selector_variance.transform(scaler.transform(X_test)),
                        columns=X_test.columns.values[selector_variance.get_support()])
    X_test.drop(corr_features, inplace=True, axis=1)
    X_test = X_test[selected_features].copy()

    num_classes = len(y_train_categorical[0])
    
    return X_train, y_train, X_test, y_test, df_test

def label_collision_data(df_features, collisions_init):
# Create a binary label column initialized to 0 (no collision)
    df_features['is_collision'] = 0

    # Iterate over each collision interval
    for collision_time in collisions_init:
        mask = (df_features['start'] <= collision_time) & (df_features['end'] >= collision_time)
        
        df_features.loc[mask, 'is_collision'] = 1

    return df_features

def predict_anomaly_score(X_test, classifier):
    try:
        anomaly_scores = classifier.predict(X_test)
        # Replace inf values with the maximum float value
        anomaly_scores = np.nan_to_num(anomaly_scores, nan=np.nanmean(anomaly_scores), posinf=np.finfo(float).max, neginf=np.finfo(float).min)
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        # If an error occurs, you might want to inspect the model's internal state
    print("Anomaly prediction completed.")
    return anomaly_scores

def find_collisions_zones(collisions):
    ts_starts = collisions[collisions['Inizio/fine'] == 'i'].Timestamp.reset_index()
    ts_ends = collisions[collisions['Inizio/fine'] == 'f'].Timestamp.reset_index()
    d = {'start': ts_starts.Timestamp, 'end': ts_ends.Timestamp}
    collision_zones = pd.DataFrame(d)
    return collision_zones