"""File description:

Script to run the experiment.
"""


import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from munitor import monitor_tic, monitor_toc

# /// Setup begin \\\
global_random_seed = 42
random_seeds_for_instance_sampling = [233, 377, 610]

# format: [dataset, [to exclude columns]]
dataset_setup_list = [
    ["IoTID20", ["Flow_ID", "Cat", "Sub_Cat"]],
    ["BoTNeTIoT-L01", ["Device_Name", "Attack", "Attack_subType"]],
    ["X-IIoTID", ["Date", "Timestamp", "class1", "class2"]],
    ["IoT-DNL", []]
]

classifiers = {
    'RF': RandomForestClassifier(),
    'DT': DecisionTreeClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'LogReg': LogisticRegression(),
    'Ridge': RidgeClassifier()
}

training_rows_sample_size_list = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]

training_features_sample_size_list = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]

# \\\ Setup end ///

results_header = ['Dataset name', 'Dataset %', 'Training sample size', 'Random seed', 'Number of features',
                  'Classifier', 'MCC', 'Training time', 'TR-CPU%', 'Testing time', 'TE-CPU%']
results = []

results_df = None

counter_model_number = 1
# dataset processing util method begin
def run(dataset):
    global results_df

    dataset_name = dataset[0]

    print(f"Started training with dataset {dataset_name}")

    # loading thr dataset
    dataset_folder = f"./datasets/{dataset_name}"

    df = pd.read_csv(f"{dataset_folder}/{os.listdir(dataset_folder)[0]}")

    # removing not useful columns
    df = df.drop(columns=dataset[1])

    # encoding all no numerical columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for column in df.columns:
        if not df[column].dtype.kind in ['i', 'f']:
            df[column] = le.fit_transform(df[column].astype(str))

    # splitting features & label
    x = df.iloc[:, :-1]
    y = df[df.columns[-1]]

    x.replace([np.inf, -np.inf], 0, inplace=True)

    # Scaling the input
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x.values)

    # splitting the dataset in train and test
    x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.33, random_state=global_random_seed)

    total_model_number = len(dataset_setup_list) * len(training_rows_sample_size_list) * 3 * len(
        training_features_sample_size_list) * len(classifiers)

    # train and test util method begin
    def train_test_report(x_train_partial, y_train_partial, current_training_features_sample_size_list, sampling_seed):
        global counter_model_number

        partial_result = []

        for training_features_sample_size in current_training_features_sample_size_list:
            # selecting features
            selector = SelectKBest(score_func=f_classif, k=training_features_sample_size)
            x_train_sample = selector.fit_transform(x_train_partial, y_train_partial)
            original_columns = set(range(x_train_partial.shape[1]))
            modified_columns = set(range(x_train_sample.shape[1]))
            removed_columns_indices = list(original_columns - modified_columns)
            x_test_sample = np.delete(x_test, removed_columns_indices, axis=1)

            for classifier_name in classifiers:
                print(
                    f"Model {counter_model_number}/{total_model_number}. Dataset {dataset_name}. Running classifier: {classifier_name} with sample size {training_rows_sample_size} and {training_features_sample_size} features. Seed {sampling_seed}")
                counter_model_number += 1

                # training
                clf = classifiers[classifier_name]
                monitor_tic()
                clf.fit(x_train_sample, y_train_partial)
                tr_action_cpu_percent, tr_action_elapsed_time = monitor_toc()

                # testing
                monitor_tic()
                y_pred = clf.predict(x_test_sample)
                te_action_cpu_percent, te_action_elapsed_time = monitor_toc()

                # saving the results
                partial_result.append([
                    training_rows_sample_size * 100,
                    len(y_train_partial),
                    sampling_seed,
                    training_features_sample_size,
                    classifier_name,
                    matthews_corrcoef(y_test, y_pred),
                    tr_action_elapsed_time,
                    tr_action_cpu_percent,
                    te_action_elapsed_time,
                    te_action_cpu_percent,
                ])


        return partial_result

    # train and test util method end

    for training_rows_sample_size in training_rows_sample_size_list:
        for sampling_seed in random_seeds_for_instance_sampling:

            if training_rows_sample_size < 1:
                x_train_partial, _, y_train_partial, _ = train_test_split(x_train, y_train,
                                                                          train_size=training_rows_sample_size,
                                                                          random_state=sampling_seed, shuffle=True,
                                                                          stratify=y_train)
            else:
                x_train_partial = x_train
                y_train_partial = y_train

            # setting the numbers of features to be used in training
            current_training_features_sample_size_list = [int(sample_size * x_train.shape[1]) for sample_size in
                                                          training_features_sample_size_list]

            # processing the dataset sample
            rep = train_test_report(x_train_partial, y_train_partial, current_training_features_sample_size_list,
                                    sampling_seed)
            for r in rep:
                results.append([dataset_name] + r)

        # saving a backup of the results file
        results_df = pd.DataFrame(results, index=None, columns=results_header)

        # Calculate means grouped by 'Seed'
        avg_df = results_df \
            .groupby(['Dataset name', 'Dataset %', 'Training sample size', 'Number of features', 'Classifier']) \
            .agg({'MCC': 'mean', 'Training time': 'mean', 'TR-CPU%': 'mean', 'Testing time': 'mean', 'TE-CPU%': 'mean'}) \
            .reset_index()
        avg_df['Random seed'] = 'AVG'
        # Append means rows to the original DataFrame
        results_df = pd.concat([results_df, avg_df], ignore_index=True)

        # Write to csv
        results_df.to_csv(f'results/backup_results_by_rows_and_features.csv', index=False)


# dataset processing util method end

for dataset_setup in dataset_setup_list:
    run(dataset_setup)

# writing the final results to csv
results_df.to_csv(f'results/results_by_rows_and_features.csv', index=False)
