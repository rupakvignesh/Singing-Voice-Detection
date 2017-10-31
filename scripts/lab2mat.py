"""
Converts label files from "start(time) end(time) label"
to a series of 0s and 1s. The duration of label is converted
to number of frames that would be expected when extracting features.
"""

import sys, os
import numpy as np
import math

gt_path = '/Users/RupakVignesh/Desktop/spring17/7100/Data/jamendo_lab/'
op_path = '/Users/RupakVignesh/Desktop/spring17/7100/Data/gt_labels/'
gt_list = sys.argv[1] 
sample_rate = 16000
hop_size = 0.008
window_size = 0.032

with open(gt_list,'r') as GT:
	gt = [lines.rstrip() for lines in GT]

labels = ['nosing', 'sing']

for i in range(len(gt)):
	with open(os.path.join(gt_path,gt[i]), 'r') as GT_FILE:
		lab_content = [lines.rstrip() for lines in GT_FILE]
	lab_content_parsed = []
	for j in range(len(lab_content)):
		lab_content_parsed.extend([str.split(lab_content[j])])
	N = int(round(float(lab_content_parsed[-1][1])*sample_rate))
	temp = (N-(window_size*sample_rate))/float(hop_size*sample_rate)
	n_zeros = int(round(sample_rate*hop_size*(np.ceil(temp)-temp)))			# Zeros for padding
	frame_wise_gt = np.zeros(int((N+n_zeros)/(hop_size*sample_rate)))		# initialize output array
	for j in range(len(lab_content_parsed)):
		ts1 = float(lab_content_parsed[j][0])					# start time stamp in sec
		ts2 = float(lab_content_parsed[j][1])					# end time stamp in sec
		frame_start = int(ts1/hop_size)						# start time in frame index
		frame_end = int(ts2/hop_size)						# end time in frame index
                label = labels.index(lab_content_parsed[j][2])				# current label
		frame_wise_gt[frame_start:frame_end] = int(label)			# alot current label to the frame indices
	File = open(os.path.join(op_path,gt[i]),'w')					# Write to file
	for j in range(len(frame_wise_gt)):
		File.write(str(frame_wise_gt[j])+'\n')
	File.close()	
