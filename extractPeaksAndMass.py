import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from random import shuffle
from parameters import extraction_options

"""
Extracts peak data from a number of samples, given a time window and mass.
Creates a folder which acts as a self contined data set, holding information about
each peak (eg retention time of the peak maximum) and the peak profile and mass
spectra at peak maximum in separate files numbered by the peak ID.

Inputs (modified from the parameters.py file):
    data_path -- Path to the '0fullResults' output folder from peak detection (https://github.com/rosalind-wang/GCPeakDetection)
    masses -- List of mass values to extract peaks from
    max_files_to_process -- The maximum number of files to extract from (int)
    max_peaks_per_file -- The maximum number of peaks to extract per file (int)
    time_window -- Tuple of (start time, end time) in minutes. Peaks are only extracted within this window.
                   Can use (0, np.inf) to get the whole chromatogram
    margin -- Number of time steps to add to either side of the time window when extracting 
    sort_by_peak_area -- If True, the peaks extracted per file are sorted by the area of the peak. Peaks with more area are extracted first
    peak_id_width -- Int specifying the number of digits in the peak id (padded by 0s).
                     Ensure that this in enough to cover the total number of peaks generated (eg 3 for up to 999 peaks)
                     so that the ordering of peaks is sorted the same across different operating systems
    save_path -- Output path, as a string
    shuffle_files -- If True, the files in the data_path are shuffled before using (up to max_files_to_process)

Outputs:
    PeakData.csv -- CSV file saved from a dataframe of shape (number of peaks extracted, 6)
                    The index of the dataframe is the peak ID which allows matching with the .txt
                    and .tsv files generated. The columns are 'PeakNumber' (within each file),
                    'startTime', 'endTime', 'peakMaxTime', 'File', 'Mass'
    WholeSequence.csv -- CSV file saved from the chromatogram segments extracted from each file.
                         The shape of the dataframe is (number of time steps, number of files).
                         The number of time steps is enough to cover the time_window, plus twice the
                         margin.
    *.txt -- Peak profiles. The values are the intensity values at each time step
             between the startTime and endTime. One file is generated per peak.
    *-MassSlice.tsv -- Intensity values along the m/z axis, taken at the peak maximum. 
                       One file is generated per peak.
    PeakProfiles-(*,*).png -- All extracted peak profiles, plotted on one figure. The file name also
                              takes note of the time_window setting used to extract the peaks
"""

### Load options
data_path = extraction_options['data_path']
masses = extraction_options['masses']
max_files_to_process = extraction_options['max_files_to_process']
max_peaks_per_file = extraction_options['max_peaks_per_file']
time_window = extraction_options['time_window']
margin = extraction_options['chromatogram_margin']
sort_by_peak_area = extraction_options['sort_by_peak_area']
w = extraction_options['peak_id_width']

save_path = extraction_options['save_path']
os.makedirs(save_path, exist_ok = True)  # Make the save_path if it doesn't exist


# Get files
files = []
for f in os.listdir(data_path):
    if f.endswith('.mat'):
        files.append(f)

if extraction_options['shuffle_files']:
    shuffle(files)
else:
    files.sort()

# Initialise
peak_id = 0
extracted_peak_data = []
chromatogram_data = []

# Extract peaks
for i in range(min(max_files_to_process, len(files))):
    file = files[i]
    
    # Load data from .mat file
    mat_data = loadmat(os.path.join(data_path, file))
    massZ = mat_data['massZ']
    resMSPeak = mat_data['resMSPeak']
#    peakData = mat_data['peakData']  -- processed peaks on full chromatogram
    dataRaw = mat_data['dataRaw']
    del mat_data
    
    for mass in masses:
        mass_index = np.flatnonzero(massZ == mass)[0]  # Find the index of the mass we want
        data = resMSPeak[mass_index][0]  # Get the appropriate data for the mass selected
        # data has the shape (n_peaks, 5), where the columns are [peak start, peak end, peak max, peak height, peak area]
        
        try:
            # Column 2 of data is peakMaxTime
            peak_window_start_index = np.flatnonzero(data[:,2] > time_window[0])[0]  # Get the index of the first peak inside the time window
            # Row 0 of dataRaw is the times
            raw_window_start_index = np.flatnonzero(dataRaw[0,:] > time_window[0])[0]  # Get the starting index of the chromatogram segment inside the time window
        except IndexError:  # Start of time window is beyond the range of the data
            chromatogram_data.append(pd.DataFrame([0], index = [0.]))  # Add a placeholder column so the files don't get misaligned (ie the column number match i)
            #Index is made to be of type float to prevent error in pd.merge_asof(direction = nearest)
            continue
        try:
            peak_window_end_index = np.flatnonzero(data[:,2] > time_window[1])[0]  # Get the index of the first peak past the time window
            raw_window_end_index = np.flatnonzero(dataRaw[0,:] > time_window[1])[0]  # Get the ending index of the chromatogram segment inside the time window
        except IndexError:  # End of time window is beyond the range of the data
            peak_window_end_index = None
            raw_window_end_index = None
        
        # Extract the chromatogram data with a margin on either side of the start and end index
        raw_window_start_index = max(0, raw_window_start_index - margin)
        if raw_window_end_index is not None:
            raw_window_end_index = raw_window_end_index + margin
        chromatogram_data.append(pd.DataFrame(dataRaw[mass_index + 1, raw_window_start_index:raw_window_end_index],
                                              index = dataRaw[0, raw_window_start_index:raw_window_end_index]))  # Index = time
        
        # Get the index values of the peaks inside the time window
        if sort_by_peak_area:
            idx = np.argsort(data[peak_window_start_index:peak_window_end_index, 4])[::-1]  # Sort the peaks by area (largest first)
            idx = idx + peak_window_start_index
        elif peak_window_end_index is not None:
            idx = range(peak_window_start_index, peak_window_end_index)
        else:
            idx = range(peak_window_start_index, data.shape[0])  # Go up to the last peak
        idx = idx[:max_peaks_per_file]  # Select only up to a maximum number of peaks
        
        # Get 'startTime', 'endTime' and 'peakMaxTime' from each peak
        start_times = data[idx, 0]
        end_times = data[idx, 1]
        peak_max_times = data[idx, 2]
        
        # Create a dataframe to hold the information about each peak
        times = np.concatenate((start_times.reshape((1,-1)), end_times.reshape((1,-1)), peak_max_times.reshape((1,-1))))
        temp_df = pd.DataFrame(times.T, columns = ['startTime','endTime','peakMaxTime'])
        temp_df['File'] = i
        temp_df['Mass'] = mass
        extracted_peak_data.append(temp_df)
        
        for j in range(len(idx)):
            # Find the start and end points of each peak to be extracted
            start_index = np.flatnonzero(dataRaw[0,:] == start_times[j])[0]
            end_index = np.flatnonzero(dataRaw[0,:] == end_times[j])[0]
            peak_max_index = np.flatnonzero(dataRaw[0,:] == peak_max_times[j])[0]
            
            extracted_peak = dataRaw[mass_index + 1, start_index:end_index+1]
            mass_slice = dataRaw[1:, peak_max_index]  # Get the mass spectra at the time of the peak maximum
            plt.plot(extracted_peak)
            np.savetxt(os.path.join(save_path, '{:0{w}d}-Mass={}-File={}-Peak={}.txt'.format(peak_id, mass, i, j, w=w)), extracted_peak, delimiter = ',')
            np.savetxt(os.path.join(save_path, '{:0{w}d}-Mass={}-File={}-Peak={}-MassSlice.tsv'.format(peak_id, mass, i, j, w=w)), mass_slice, delimiter = '\t')
            peak_id += 1


# Create a dataframe to hold all information and save to file
df = pd.concat(extracted_peak_data)
df.reset_index(inplace = True)
df.columns = ['PeakNumber', 'startTime', 'endTime', 'peakMaxTime', 'File', 'Mass']
df.to_csv(os.path.join(save_path, 'PeakData.csv'))
plt.savefig(os.path.join(save_path, 'PeakProfiles-{}.png'.format(time_window)), dpi = 300, format = 'png')


# Create af dataframe of the chromatogram segments data, making sure that the time index matches up
# To do this the dataframes are merged using a pandas asof merge, which matches on the nearest key
# The longest dataframe is used first since it has the most time points and the left_index is
# the one that remains as the index of the output dataframe
sort_index = sorted(range(len(chromatogram_data)), key = lambda x: len(chromatogram_data[x]), reverse = True)

df_chromatogram = chromatogram_data[sort_index[0]]
for i, idx in enumerate(sort_index):
    if i == 0:
        continue
    # Join the next dataframe to df_chromatogram using the left index and right index
    df_chromatogram = pd.merge_asof(df_chromatogram, chromatogram_data[idx], direction = 'nearest', left_index = True, right_index = True)
# A “nearest” search selects the row in the right DataFrame whose ‘on’ key is closest in absolute distance to the left’s key.
# Using direction = "nearest" also fills any blank values at the start with the first available value and 
# any blank values at the end with the last available value

# Rearrange the columns back to the order of the files processed (to match up with the 'File' number saved in PeakData.csv)
df_chromatogram.columns = sort_index
df_chromatogram.sort_index(axis = 1, inplace = True)
df_chromatogram.to_csv(os.path.join(save_path, 'WholeSequence.csv'))
