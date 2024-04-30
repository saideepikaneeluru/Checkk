"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
from numpy.typing import NDArray
import pickle
import matplotlib.backends.backend_pdf as pdf

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def plot_clustering(data, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

def compute_SSE(data, labels):
  
    sse = 0.0
    for i in np.unique(labels):
        cluster_points = data[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - cluster_center) ** 2)
    return sse

def adjusted_rand_index(true_labels, pred_labels):
 
    contingency_matrix = np.zeros((np.max(true_labels) + 1, np.max(pred_labels) + 1), dtype=np.int64)
    for i in range(len(true_labels)):
        contingency_matrix[true_labels[i], pred_labels[i]] += 1

    a = np.sum(contingency_matrix, axis=1)
    b = np.sum(contingency_matrix, axis=0)
    n = np.sum(contingency_matrix)
    ab_sum = np.sum(a * (a - 1)) / 2
    cd_sum = np.sum(b * (b - 1)) / 2
    a_plus_b_choose_2 = np.sum(a * (a - 1)) / 2
    c_plus_d_choose_2 = np.sum(b * (b - 1)) / 2

    ad_bc = np.sum(contingency_matrix * (contingency_matrix - 1)) / 2

    expected_index = a_plus_b_choose_2 * c_plus_d_choose_2 / n / (n - 1) + ad_bc ** 2 / n / (n - 1)
    max_index = (a_plus_b_choose_2 + c_plus_d_choose_2) / 2
    return (ad_bc - expected_index) / (max_index - expected_index)



def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    k = params_dict['k']  # Number of neighbors to consider
    s_min = params_dict['s_min']   # Similarity threshold
    
    n = len(data)
    computed_labels = np.zeros(n, dtype=np.int32)

    for i in range(n):
        # Compute distances from the current data point to all other points
        distances = cdist([data[i]], data, metric='euclidean')[0]

        # Sort distances and get the indices of k nearest neighbors
        nearest_indices = np.argsort(distances)[1:k+1]  # Exclude the current point itself

        # Count labels of nearest neighbors
        label_counts = np.bincount(labels[nearest_indices])

        # Get the label with the highest count
        majority_label = np.argmax(label_counts)

        # Compute similarity between the current point and its nearest neighbor
        similarity = label_counts[majority_label] / k

        # Assign cluster label if similarity meets the threshold
        if similarity >= s_min:
            computed_labels[i] = majority_label + 1  # Add 1 to avoid 0 as a label

  
    # Calculate ARI
    ARI = adjusted_rand_index(labels,computed_labels)

    # Calculate SSE manually
    SSE = compute_SSE(data,computed_labels)

    return computed_labels, SSE, ARI

def hyperparameter_study(data, labels, k_range, s_min_range, num_trials):
    best_ARI = -1
    best_k = None
    best_s_min = None

    for k in k_range:
        for s_min in s_min_range:
            total_ARI = 0
            for _ in range(num_trials):
                params_dict = {'k': k, 's_min': s_min}
                computed_labels, ARI, _ = jarvis_patrick(data, labels, params_dict)
                total_ARI += ARI

            avg_ARI = total_ARI / num_trials
            if avg_ARI > best_ARI:
                best_ARI = avg_ARI
                best_k = k
                best_s_min = s_min

    return best_k, best_s_min

def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').
    
    cluster_data = np.load('question1_cluster_data.npy')
    cluster_labels = np.load('question1_cluster_labels.npy')

    random_indices = np.random.choice(len(cluster_data), size=5000, replace=False)
    data_subset = cluster_data[random_indices]
    labels_subset = cluster_labels[random_indices]
    
    data_subset = data_subset[:1000]
    labels_subset = labels_subset[:1000]
    
    # Perform hyperparameter study
    k_range = [3,4,5,6,7,8]
    s_min_range = [0.4, 0.5, 0.6, 0.7, 0.8,0.9,1.0]

    num_trials = 10

    best_k, best_s_min = hyperparameter_study(data_subset, labels_subset, k_range, s_min_range, num_trials)
    
    params_dict = {'k': best_k, 's_min': best_s_min}
    
    groups = {}
    plots_values={}
    # Apply best hyperparameters on five slices of data
    for i in [0,1,2,3,4]:
        data_slice = cluster_data[i * 1000: (i + 1) * 1000]
        labels_slice = cluster_labels[i * 1000: (i + 1) * 1000]
        # Apply best hyperparameters on the slice of data
        computed_labels, sse,ari = jarvis_patrick(data_slice, labels_slice, params_dict)
        groups[i] = {"smin": best_s_min,"k":best_k, "ARI": ari, "SSE": sse}
        plots_values[i] = {"computed_labels": computed_labels, "ARI": ari, "SSE": sse}
        
    highest_ari = -1
    best_dataset_index = None
    for i, group_info in plots_values.items():
        if group_info['ARI'] > highest_ari:
            highest_ari = group_info['ARI']
            best_dataset_index = i
            
    pdf_pages = pdf.PdfPages("jarvis_patrick_clustering_plots.pdf")
    
    
    plt.figure(figsize=(8, 6))
    plot_ARI = plt.scatter(cluster_data[best_dataset_index * 1000: (best_dataset_index + 1) * 1000, 0], 
                cluster_data[best_dataset_index * 1000: (best_dataset_index + 1) * 1000, 1], 
                c=plots_values[best_dataset_index]["computed_labels"], cmap='viridis')
    plt.title(f'Clustering for Dataset {best_dataset_index} (Highest ARI) with k value :{best_k} and s_min: {best_s_min}')
    plt.suptitle('Jarvis - Patrick Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    pdf_pages.savefig() 
    plt.close()
    
    lowest_sse = float('inf')
    best_dataset_index_sse = None
    for i, group_info in plots_values.items():
        if group_info['SSE'] < lowest_sse:
            lowest_sse = group_info['SSE']
            best_dataset_index_sse = i
            
    plt.figure(figsize=(8, 6))
    plot_SSE = plt.scatter(cluster_data[best_dataset_index_sse * 1000: (best_dataset_index_sse + 1) * 1000, 0], 
                cluster_data[best_dataset_index_sse * 1000: (best_dataset_index_sse + 1) * 1000, 1], 
                c=plots_values[best_dataset_index_sse]["computed_labels"], cmap='viridis')
    plt.title(f'Clustering for Dataset {best_dataset_index_sse} (Lowest SSE) with k value :{best_k} and s_min: {best_s_min}')
    plt.suptitle('Jarvis - Patrick Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    pdf_pages.savefig()  
    plt.close()
    
    pdf_pages.close()
    
    # Create a dictionary for each parameter pair ('sigma' and 'xi').
 

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"] #{}
    

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    #plot_ARI = plt.scatter([1,2,3], [4,5,6])
    #plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    ari_values = [group_info["ARI"] for group_info in groups.values()]
    mean_ari = np.mean(ari_values)
    std_dev_ari = np.std(ari_values)
    
    # A single float
    answers["mean_ARIs"] = mean_ari

    # A single float
    answers["std_ARIs"] = std_dev_ari
    
    sse_values = [group_info["SSE"] for group_info in groups.values()]
    mean_sse = np.mean(sse_values)
    std_dev_sse = np.std(sse_values)

    # A single float
    answers["mean_SSEs"] = mean_sse

    # A single float
    answers["std_SSEs"] = std_dev_sse

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
