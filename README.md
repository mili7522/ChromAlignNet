# ChromAlignNet

Implementation of "Peak alignment of gas chromatography–mass spectrometry data with deep learning" - https://www.sciencedirect.com/science/article/pii/S0021967319308623.

## Outline:
This model uses a deep neural network for the alignment of Gas Chromatography - Mass Spectroscopy (GC-MS) peaks.
It utilises a siamese network architecture to compare the mass spectra, chromatogram and information extracted about each peak, such as the retention time corresponding to maximum intensity.

It provides good alignment results for complex data sets, such as the human breath.


## Requirements:
The scripts have been tested under Python 3.6.1, with the following packages installed (along with their dependencies):

* tensorflow==1.8.0
* keras==2.1.3

## Instructions:
Preprocessed GC-MS data, saved models and prediction outcomes can be found at https://doi.org/10.25919/5ca16f2db73a9

Models are trained and saved using TrainChromAlignNet.py and predictions are made using the saved models with PredictChromAlignNet.py. The alignment outcomes can be visualised with plotResults.py. Various parameters to control the training and prediction process is set in parameters.py.

To train the model using new data, peaks need to be first extracted using the peak detection algorithm provided at https://github.com/rosalind-wang/GCPeakDetection.


## Contact:
Mike at mili7522@uni.sydney.edu.au
