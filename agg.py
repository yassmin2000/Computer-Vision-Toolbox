



import cv2 as cv
import cv2
import numpy as np
from PyQt5.QtWidgets import QSlider
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

def eculidean_distance(point1,point2):

    """
    Calculates the Euclidean distance between two points.

    Parameters:
    point1 (numpy.ndarray): The first point.
    point2 (numpy.ndarray): The second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((point1-point2)**2))

def extract_features(img):

    """
    Reshapes the input image to a 2D array with 3 features (RGB) and converts it to float32 format.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The reshaped and converted image.
    """

    new_img = img.reshape((-1,3)) ## 3 features RGB
    new_img = new_img.astype("float32")
    return new_img

def init_clusters(features,nump_int):

    """
    Initializes clusters based on the given features and number of clusters.

    Parameters:
    features (list): The features to be clustered.
    num_clusters (int): The number of clusters to be created.

    Returns:
    list: The initialized clusters.
    """
    cluster_color = int(256 / nump_int)
    groups = {}
    for i in range(nump_int):
        color = i * cluster_color
        groups[(color, color, color)] = []
        
    for pixel_point in features:
        min_dist = float('inf')
        for key ,group in groups.items():
            dist = eculidean_distance(pixel_point,list(key))
            if dist< min_dist:
                min_dist = dist
                target_cluster = key
        groups[target_cluster].append(pixel_point)  
    print(f"length of clusters:{len(list(groups.keys()))}")    
    return [group for group in groups.values() if len(group) > 0]

def centroid_linkage(cluster1,cluster2):

    """
    Calculates the Euclidean distance between the centroids of two clusters.

    Parameters:
    cluster1 (numpy.ndarray): The first cluster.
    cluster2 (numpy.ndarray): The second cluster.

    Returns:
    float: The Euclidean distance between the centroids of the two clusters.
    """
    mean_cluster1 =np.mean(cluster1,axis=0)
    mean_cluster2 =np.mean(cluster2,axis=0)
    return eculidean_distance(mean_cluster1,mean_cluster2)

def find_the_closest_clusters(clusters_list):
    """
    Finds the two clusters in the given list that are closest to each other based on the centroid linkage distance.

    Parameters:
    clusters_list (list): The list of clusters.

    Returns:
    tuple: The indices of the two closest clusters.
    """
    min_dist = float('inf')
    for cluster1_idx in range(len(clusters_list[:-1])):
        for cluster2_idx in range((cluster1_idx+1),len(clusters_list)):
            dist = centroid_linkage(clusters_list[cluster1_idx],clusters_list[cluster2_idx])
            if dist <  min_dist:
                min_dist = dist
                closest_clusters = (cluster1_idx,cluster2_idx)
    return closest_clusters


def merge_clusters(closest_clusters,clusters_list):
    """
    Merges the two closest clusters in the given list of clusters.

    Parameters:
    closest_clusters (tuple): The indices of the two closest clusters.
    clusters_list (list): The list of clusters.

    Returns:
    list: The updated list of clusters with the two closest clusters merged.
    """
    new_clusters_list = []
    new_clusters_list.append(clusters_list[closest_clusters[0]] + clusters_list[closest_clusters[1]])
    for cluster_id in range(len(clusters_list)):
        if cluster_id == closest_clusters[0] or cluster_id == closest_clusters[1]:
            continue
        new_clusters_list.append(clusters_list[cluster_id])
    return new_clusters_list



def apply_agglomerative_clustering(image,K_agg ,nump_int): 

    """
    Applies agglomerative clustering to the given image.

    Parameters:
    image (numpy.ndarray): The input image.
    K_agg (int): The desired number of clusters.
    num_points_int (int): The number of points to initialize clusters.

    Returns:
    dict: A dictionary mapping points to cluster numbers.
    dict: A dictionary mapping cluster numbers to cluster centers.
    """

    cluster = {}
    centers = {}
    features = extract_features(image)
    clusters_list = init_clusters(features,nump_int) 
    while K_agg < len(clusters_list):
        closest_clusters = find_the_closest_clusters(clusters_list)
        print(f"closest_clusters:{closest_clusters}")
        clusters_list = merge_clusters(closest_clusters,clusters_list)
        print(f"K_agg:{len(clusters_list) }")
    for cl_num, cl in enumerate(clusters_list):
        for point in cl:
            cluster[tuple(point)] = cl_num
            
    for cl_num, cl in enumerate(clusters_list):
        centers[cl_num] = np.mean(cl, axis=0)
   
    return cluster,centers
def get_cluster_center(point, cluster, centers):
    
    """
    Returns the center of the cluster to which the given point belongs.
    """
    point_cluster_num = cluster[tuple(point)]
    center = centers[point_cluster_num]
    return center

def get_segmentated_image(image, cluster,centers):
    """
    Returns the center of the cluster to which the given point belongs.

    Parameters:
    point (tuple): The coordinates of the point.
    cluster (dict): A dictionary mapping points to cluster numbers.
    centers (dict): A dictionary mapping cluster numbers to cluster centers.

    Returns:
    numpy.ndarray: The center of the cluster to which the point belongs.
    """
    output_image = []
    for row in image:
        rows = []
        for col in row:
            rows.append(get_cluster_center(list(col), cluster, centers))
        output_image.append(rows)    
    output_image = np.array(output_image, np.uint8)
    return output_image