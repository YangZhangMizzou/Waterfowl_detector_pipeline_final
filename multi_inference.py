import os
import glob
from datetime import date

today = date.today()
det_list = ['fasterrcnn','yolonas','retinanet','retinanetknn','yolo5']
cla_list = ['res18','mixmatch']
dataset = 'Bird_I_Test_habitat'
dataset_root = 'example_images'
all_folders_dataset = glob.glob(os.path.join(dataset_root,dataset,'test','ShrubScrub'))
csv_root = os.path.join(dataset_root,dataset,'image_info.csv')
for folder in all_folders_dataset:
	for det_model in det_list:
		for cla_model in cla_list:
			height = folder.split(os.sep)[-1]
			subset = folder.split(os.sep)[-3]
			save_dir = os.path.join('Result',str(today),subset,det_model,cla_model,height)
			os.system('python inference_image_height.py --image_root {}  --out_dir {} --evaluate True --det_model {} --cla_model {} --csv_root {} --image_ext {}'.format(folder,save_dir,det_model,cla_model,csv_root,'jpg'))