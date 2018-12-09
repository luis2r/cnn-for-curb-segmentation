#!/usr/bin/python

import os

# start editable vars #
outputfile	= "~/trainin_bev_kitti_car_people_bici.txt"	# file to save the results to
folder		= "/home/shared/datasets/bird_eye_view/kitti/data"		# the folder to inventory
exclude		= ['Thumbs.db','.tmp']	# exclude files containing these strings
pathsep		= "/"			# path seperator ('/' for linux, '\' for Windows)
# end editable vars #

with open(outputfile, "w") as txtfile:
	for path,dirs,files in os.walk(folder):
		sep = "\n---------- " + path.split(pathsep)[len(path.split(pathsep))-1] + " ----------"
		print (sep)
		txtfile.write("%s\n" % sep)

		for fn in sorted(files):
			if not any(x in fn for x in exclude):
				filename = os.path.splitext(fn)[0]
				
				print (filename)
				txtfile.write("%s\n" % filename)

txtfile.close()