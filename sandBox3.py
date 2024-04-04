def train_eval_gcnae(graph_path, batch_size=0, attribute_dims=None):
    import warnings
    warnings.simplefilter(action='ignore')

    import pandas as pd
    import numpy as np
    import torch
    from pygod.models import GCNAE
    from pygod.utils import data_loader
    from pygod.metric import eval_roc_auc, recall_at_k
    from pythresh.thresholds.iqr import IQR
    from sklearn.metrics import precision_recall_curve, auc, f1_score
    from torch_geometric.transforms import GCNNorm
    from tqdm import tqdm
    import time
    import os

    start = time.time()

    if not os.path.exists('results'):
        os.makedirs('results')

    transform = GCNNorm()

    # Initialize lists to store the evaluation metrics
    auc_roc_list = []
    auc_pr_list = []
    r_10_list = []
    f1_list = []
    hit_rate_list = []
    recall_at_10_list_per_type = []

    files = [f for f in os.listdir(graph_path) if os.path.isfile(os.path.join(graph_path, f))]

    for filename in tqdm(files):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        file_path = os.path.join(graph_path, filename)

        data_loader_instance = data_loader(file_path, device, transform)

        model = GCNAE(device=device, encoder_layers=2, decoder_layers=1, attribute_dims=attribute_dims)
        model.fit(data_loader_instance, batch_size=batch_size)

        decision_scores = model.decision_score_
        thres = IQR()
        labels = thres.eval(decision_scores)

        # Metrics
        auc_roc = eval_roc_auc(data_loader_instance.get_labels().cpu().numpy(), decision_scores)
        precision, recall, _ = precision_recall_curve(data_loader_instance.get_labels().cpu().numpy(), decision_scores)
        auc_pr = auc(recall, precision)
        recall_at_10 = recall_at_k(data_loader_instance.get_labels().cpu().numpy(), decision_scores, 10)
        f1 = f1_score(data_loader_instance.get_labels().cpu().numpy(), labels)

        # Append the evaluation metrics to the corresponding lists
        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)
        r_10_list.append(recall_at_10)
        f1_list.append(f1)

        # Add predicted binary labels to the DataFrame
        nodes_y_df = data_loader_instance.get_node_y_df()
        nodes_y_df["predicted_binary"] = labels
        nodes_y_df["correct"] = ((nodes_y_df["predicted_binary"] == 1) & (nodes_y_df["original_y"] > 0)) | (
                    (nodes_y_df["predicted_binary"] == 0) & (nodes_y_df["original_y"] == 0))

        # Calculate hit rates for each original value and store them in a dictionary
        hit_rates = nodes_y_df.groupby("original_y")["correct"].mean().to_dict()
        hit_rate_list.append(hit_rates)

        # Compute Recall @ 10 for each type
        # Step 1: Determine the threshold for the top 10%
        threshold = np.percentile(decision_scores.cpu().numpy(), 90)

        # Step 2: Use the threshold to set positive/negative predictions
        predicted_binary = (decision_scores >= threshold).astype(int)

        # Step 3: Compute Recall for each class
        recalls_for_each_class = {}

        original_y = data_loader_instance.get_labels().cpu().numpy()
        unique_classes = np.unique(original_y)  # Get the unique classes
        for cls in unique_classes:
            true_positives = np.sum((original_y == cls) & (predicted_binary == 1))
            actual_positives = np.sum(original_y == cls)
            recall = true_positives / (actual_positives + 1e-10)  # Add a small value to prevent division by zero
            recalls_for_each_class[cls] = recall

        recall_at_10_list_per_type.append(recalls_for_each_class)

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
    results_df.to_csv('results/' + os.path.basename(graph_path).strip('/').split('/')[-1] + '_gcnae_results.csv', index=False)
    hit_rate_df.to_csv('results/' + os.path.basename(graph_path).strip('/').split('/')[-1] + '_gcnae_hitrate.csv', index=False)

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
