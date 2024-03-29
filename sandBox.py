from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch([{"host": "localhost", "port": 9200}])  # Update host and port if needed

def train_eval_ae(tab_path, batch_size=8192, attribute_dims=None):
    import warnings
    warnings.simplefilter(action='ignore')

    import pandas as pd
    import numpy as np

    from .training_utils import recall_at_k, Dataset
    from models.AE import AE

    import gc
    import torch
    from sklearn.preprocessing import StandardScaler
    from pygod.metric import eval_roc_auc
    from pythresh.thresholds.iqr import IQR
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    from sklearn.metrics import f1_score

    from torch_geometric import data

    import os
    from os import listdir
    from os.path import isfile, join
    import pickle
    from tqdm import tqdm

    import time

    start = time.time()

    if not os.path.exists('results'):
        os.makedirs('results')

    # Initialize lists to store the evaluation metrics
    auc_roc_list = []
    auc_pr_list = []
    r_10_list = []
    f1_list = []
    hit_rate_list = []
    recall_at_10_list_per_type = []  # Initialize a list to store the Recall@10 per type


    files = [f for f in listdir(tab_path) if isfile(join(tab_path, f))]

    for filename in tqdm(files):
        file_path = tab_path + filename

        # Read data in chunks to avoid memory overload
        chunksize = 5000  # Adjust chunk size based on your memory limitations
        for chunk in pd.read_csv(file_path, dtype={'exec_id': int, 'y': int}, chunksize=chunksize):
        # Preprocess data in chunks
            chunk = pd.read_csv(file_path, dtype={'exec_id': int, 'y': int})
            chunk = chunk.sort_values(by=['exec_id', 'elapsed_time'])
            original_y = chunk[['event_id', 'exec_id', 'y']]
            chunk.y = (chunk.y > 0).astype(int)
            X = chunk.drop(columns=['event_id', 'elapsed_time', 'y'])
            X = X.pivot_table(index='exec_id', columns=X.groupby('exec_id').cumcount())
            X = X.sort_index(axis='columns', level=1)
            X.columns = X.columns.map('{0[0]}|{0[1]}'.format)
            X_mask = X.isnull()
            X = X.fillna(0)

        # Convert DataFrame to dictionary and store in Elasticsearch
        data = X.to_dict(orient='records')
        for item in data:
            item['mask'] = X_mask.to_dict(orient='records')[0]  # Include mask for later processing
            es.index(index="anomaly_data", body=item)

    # Retrieve data from Elasticsearch for training
    query = {
        "size": 5000,  # Adjust size based on your needs and expected data volume
        "query": {
            "match_all": {}
        }
    }

    data = []
    scroll_response = es.search(index="anomaly_data", scroll='1m', body=query)  # Scroll through results in chunks
    scroll_id = scroll_response['_scroll_id']
    scroll_size = scroll_response['hits']['hits']

    while scroll_size > 0:
        # Process data in chunks retrieved from Elasticsearch
        for item in scroll_response['hits']['hits']:
            data.append(item['_source'])
        scroll_response = es.scroll(scroll_id=scroll_id, scroll='1m')
        scroll_id = scroll_response['_scroll_id']
        scroll_size = scroll_response['hits']['hits']

    # Convert retrieved data to pandas DataFrame
    df = pd.DataFrame(data)

    # Prepare Dataset Object
    dataset = Dataset(X, X.iloc[:, :1], df.groupby('exec_id').size().max(), attribute_dims, X_mask)

        # Instantiate AE class
    ae = AE()

        # Construct the model with your dataset
    model, features, _ = ae.model_fn(dataset, **ae.config)

        # Train the model
    model.fit(features, features, epochs=100, batch_size=batch_size, verbose=False)

        # Assign the trained model to the DAE instance
    ae.model = model

        # Now you can use your trained model to detect anomalies
    decision_scores = ae.detect(dataset)

        # Apply threshold
    thres = IQR()
    labels = thres.eval(decision_scores)

        # Metrics
    auc_roc = eval_roc_auc(df.y, decision_scores)
    precision, recall, _ = precision_recall_curve(df.y, decision_scores)
    auc_pr = auc(recall, precision)
    recall_at_10 = recall_at_k(df.y, decision_scores, 10)
    f1 = f1_score(df.y, labels)

        # Append the evaluation metrics to the corresponding lists
    auc_roc_list.append(auc_roc)
    auc_pr_list.append(auc_pr)
    r_10_list.append(recall_at_10)
    f1_list.append(f1)

        # Add predicted binary labels to the DataFrame
    original_y["predicted_binary"] = labels
    original_y["correct"] = ((original_y["predicted_binary"] == 1) & (original_y["y"] > 0)) | (
                (original_y["predicted_binary"] == 0) & (original_y["y"] == 0))

        # Calculate hit rates for each original value and store them in a dictionary
    hit_rates = original_y.groupby("y")["correct"].mean().to_dict()
    hit_rate_list.append(hit_rates)

       # Compute Recall @ 10 for each type
        # Step 1: Determine the threshold for the top 10%
    threshold = np.percentile(decision_scores, 90)

        # Step 2: Use the threshold to set positive/negative predictions
    predicted_binary = (decision_scores >= threshold).astype(int)

        # Step 3: Compute Recall for each class
    recalls_for_each_class = {}

    original_y_values = original_y['y'].values
    predicted_binary = np.array(predicted_binary)

    unique_classes = np.unique(original_y_values)  # Get the unique classes
    for cls in unique_classes:
        true_positives = np.sum((original_y_values == cls) & (predicted_binary == 1))
        actual_positives = np.sum(original_y_values == cls)
        recall = true_positives / (actual_positives + 1e-10)  # Add a small value to prevent division by zero
        recalls_for_each_class[cls] = recall


    recall_at_10_list_per_type.append(recalls_for_each_class)

    gc.collect()

    results_dict = {'F1': f1_list,
                    'AUC ROC': auc_roc_list,
                    'AUC Precision-Recall': auc_pr_list,
                    'Recall @ 10': r_10_list
                    }

    results_df = pd.DataFrame(results_dict)

    results_mean = results_df.mean()
    results_std = results_df.std()

    # Compute mean and standard deviation of hit rates
    hit_rate_df = pd.DataFrame(hit_rate_list)
    hit_rate_mean = hit_rate_df.mean()
    hit_rate_std = hit_rate_df.std()

    # Compute mean and standard deviation of hit rates
    recall_at_10_df_per_type = pd.DataFrame(recall_at_10_list_per_type)
    recall_at_10_mean_per_type = recall_at_10_df_per_type.mean()
    recall_at_10_std_per_type = recall_at_10_df_per_type.std()

    # Print the results as a table
    print('Evaluation metrics:')
    print('-------------------')
    print('{:<25s} {:<10s} {:<10s}'.format('', 'Mean', 'Std'))
    for col in results_df.columns:
        print('{:<25s} {:<10.1f} {:<10.1f}'.format(col, results_mean[col] * 100, results_std[col] * 100))

    # Save the results to a CSV file
    results_df.to_csv('results/' + tab_path.strip('/').split('/')[-1] + '_ae_results.csv', index=False)
    hit_rate_df.to_csv('results/' + tab_path.strip('/').split('/')[-1] + '_ae_hitrate.csv', index=False)

    # Print hit rate mean and standard deviation
    print('')
    for col in hit_rate_df.columns:
        print("Hit Rate {:.0f}: {:.2f} ± {:.2f}".format(col, hit_rate_mean[col] * 100, hit_rate_std[col] * 100))

    # Print Recall @ 10 mean and standard deviation per type
    print("\nRecall @ 10 for each type:")
    for col in recall_at_10_df_per_type.columns:
        print("Recall @ 10 for {:.0f}: {:.2f} ± {:.2f}".format(col, recall_at_10_mean_per_type[col] * 100, recall_at_10_std_per_type[col] * 100))

    end = time.time()
    print('')
    print('{:<25s} {:<10.1f} {:<10.1f}'.format('Time: Total / Average', end - start, (end - start) / 10))


