import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.cluster


### Group and cluster
def getDistances(prediction):
    """
    Turns a vector of probabilities into a vector of distances
        
    Arguments:
        prediction -- Numpy array giving a column vector of probabilities (from main output of ChromAlignNet)
    
    Returns:
        distances -- Numpy array giving a column vector of distances
    """
    distances = 1 / prediction
    return distances
    

def getDistanceMatrix(comparisons, number_of_peaks, prediction, clip = 10, info_df = None):
    """
    Produces a matrix of distances between all peaks
        
    Arguments:
        comparisons -- Numpy array with two columns - x1 and x2 - containing the IDs of the two peaks being compared
        number_of_peaks -- Int: Total number of peaks under consideration
        prediction -- Numpy array giving a column vector of probabilities (from main output of ChromAlignNet)
        clip -- Int or Float: Maximum value of the distance matrix
        info_df -- DataFrame containing information about each peak, in particular the peak times and file number
    
    Returns:
        distance_matrix -- 2D numpy array of distances between all peaks
    """
    distances = getDistances(prediction)
    
    distance_matrix = np.empty((number_of_peaks, number_of_peaks))
    distance_matrix.fill(clip)  # Clip value
    
    probability_matrix = np.zeros((number_of_peaks, number_of_peaks))
    
    for i, (x1, x2) in enumerate(comparisons):
        if info_df is not None and info_df.loc[x1, 'File'] == info_df.loc[x2, 'File']:
            val = min(distances[i] * 2, clip)
        elif info_df is not None and np.abs(info_df.loc[x1, 'peakMaxTime'] - info_df.loc[x2, 'peakMaxTime']) > 0.5:
            val = min(distances[i] * 2, clip)
        else:
            val = min(distances[i], clip)
        distance_matrix[x1, x2] = distance_matrix[x2, x1] = val
        probability_matrix[x1, x2] = probability_matrix[x2, x1] = prediction[i]
    
    for i in range(number_of_peaks):
        distance_matrix[i,i] = 0
        probability_matrix[i,i] = 1
    
    return distance_matrix, probability_matrix


def assignGroups(distance_matrix, threshold = 2, plot_dendrogram = False):
    """
    Assigns a group number to peaks based on what should be aligned together
    
    Arguments:
        distance_matrix -- 2D numpy array of distances between all peaks
        threshold -- Int or Float: Value to cut the dendrogram to obtain the clusters
        plot_dendrogram -- Boolean: To create a new figure for a plot of the dendrogram
    
    Returns:
        groups -- Dictionary with group ID as keys and sets of peak IDs for each group
    """
    sqform = scipy.spatial.distance.squareform(distance_matrix)
    mergings = scipy.cluster.hierarchy.linkage(sqform, method = 'average')
    if plot_dendrogram:
        plt.figure()
        scipy.cluster.hierarchy.dendrogram(mergings, leaf_font_size = 3, color_threshold = threshold)
    labels = scipy.cluster.hierarchy.fcluster(mergings, threshold, criterion = 'distance')
    
    groups = {}
    for i in range(max(labels)):
        groups[i] = set(np.where(labels == i + 1)[0])  # labels start at 1
    
    return groups

def postprocessGroups(groups, info_df):
    """
    Adjusts the groups assigned to each peak
    
    Arguments:
        groups -- Dictionary with group ID as keys and sets of peak IDs for each group
        info_df -- DataFrame containing information about each peak, in particular the peak times and file number
    
    Returns:
        new_groups -- Dictionary with new group ID as keys and sets of peak IDs for each group
    """
    max_group = len(groups) - 1  # max(groups.keys())
    new_groups = {}
    for i, group in groups.items():
        group_df = info_df.loc[group].copy()
        group_df.sort_values(by = 'peakMaxTime', axis = 0, inplace = True)
        files = group_df['File']
        files_count = dict()
        new_groups[i] = set()
        max_group_increment = 0
        for peak, file in files.iteritems():
            if file in files_count:
                if max_group + files_count[file] not in new_groups:
                    new_groups[max_group + files_count[file]] = set()
                new_groups[max_group + files_count[file]].add(peak)
                if files_count[file] > max_group_increment:
                    max_group_increment += 1
                files_count[file] += 1
            else:
                files_count[file] = 1
                new_groups[i].add(peak)
                
        max_group += max_group_increment
    
    return new_groups
            

def alignTimes(groups, info_df, peak_intensity, align_to):
    """
    Creates a column in the info_df Dataframe with the aligned times of all peaks.
    Aligns the peaks in each group according to the average time of their peakMaxTime, weighted by the peak intensity
    
    Arguments:
        groups -- Dictionary with group ID as keys and sets of peak IDs for each group
        info_df -- DataFrame containing information about each peak, in particular the peak times
        peak_intensity -- pandas Series of peak intensities
        align_to -- String: Giving the name of the new column in the info_df Dataframe which will contain the aligned times
    """
    info_df[align_to] = info_df['peakMaxTime']
    for group in groups.values():
        times = info_df.loc[group, 'peakMaxTime']
        peak_values = peak_intensity.loc[group]
        average_time = np.average(times, weights = peak_values)
#        average_time = np.mean(times)
        info_df.loc[group, align_to] = average_time