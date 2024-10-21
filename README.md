# AlzheimerAndFrontotemporalAnalysis
Detection of Alzheimer's and Frontotemporal Dementia by Machine Learning Using EEG Signals

- In this study, EEG signals obtained from Alzheimer's (AD), Frontotemporal dementia (FTD) and healthy age-matched individuals (C) were studied. 
- The dataset consists of EEG recordings from 88 individuals (36 AD, 23 FTD, 29 C). 
- The dataset is publicly available under the name OpenNeuroDatasets/ds004504. In the dataset, the signals are under two folders, processed and unprocessed. Since the signals were already processed, they were used in the project without the need for any further processing.

- A Butterworth filter with a frequency range of 0.5 to 45 Hz is applied to the signals. 
- Signals were then subjected to the Artifact Subspace Reconstruction (ASR) routine, an artifact rejection technique, and bad data periods exceeding 17, a maximum window standard deviation of 0.5 s, were removed. 
- The ICA method (RunICA algorithm) was then used to convert 19 EEG channels into 19 ICA components. 
- Signals are epoched to 4-second time windows.
- The Relative Band Power (RBP), the Power Spectral Density (PSD) of the time-window signal for each frequency band, was obtained using the Welch method, which divides the signal into segments and calculates the quadratic magnitude of the discrete Fourier transform of each segment. 
- Finally, the relative ratio of the PSD of each band was calculated for each epoch. To calculate the relative ratio of the PSD of a band, the PSD of the band is calculated and then divided by the PSD of the entire frequency range of interest, i.e., 0.5-45 Hz. 
- Absolute Band power (ABP) is calculated. 

