import cv2
import numpy
import glob

root_dir = './dataset/'
for height_folder in glob.glob(root_dir+'*/'):
	for txt_dir in glob.glob(height_folder+'/*.txt'):
		image_name = txt_dir.split('/')[-1].split('.')[0]
		mask_dir = height_folder+'{}/predict.png'.format(image_name)
		cv2.imread(image_name)
		print()