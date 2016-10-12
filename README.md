# EyeTracking
MATLAB code for processing and modelling eye tracking data

Currently includes code for processing eye tracking data only. Files are:  

* process_file_script.m:	MATLAB script that takes one eye tracking file as input, and
						returns 2 cells, one containing the time stamps and x-y 
						coordinates of the eye tracking data, and another containing 
						the same for the target data

* controlSeq.m:			MATLAB function that takes the two cells output by the 
						process_file_script.m script and returns a cell containing the
						target coordinates for each time stamp in each eye tracking 
						series

* loadPatients: takes a directory of patients and loads all of them in memory. The following events and data are parsed: 
all the trials with eye tracking, pupil dilatation, blinks and saccades detected by the system; the size of the screen;
the name of the patient (if it is included in the namefile)
* preprocessPatient.m: Takes a file with one patient and does the parsing of all the events.
* estimateGaps: Takes a data structure of patients parsed by loadPatients and tries to estimate the gaps using a Gaussian Process.
The parameters of the model are estimated using the Type II MAP. The module for the GP depends on the library by 