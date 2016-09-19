# EyeTracking
MATLAB code for processing and modelling eye tracking data

Currently includes code for processing eye tracking data only. Files are:

process_file_script.m:	MATLAB script that takes one eye tracking file as input, and
						returns 2 cells, one containing the time stamps and x-y 
						coordinates of the eye tracking data, and another containing 
						the same for the target data

controlSeq.m:			MATLAB function that takes the two cells output by the 
						process_file_script.m script and returns a cell containing the
						target coordinates for each time stamp in each eye tracking 
						series