import os
import glob
from datetime import date

#dataset
# today = date.today()
# model_list = ['yolonas','faster','yolo']
# dataset_root = './example_images/drone_collection_dataset/test'
# dataset_list = ['G','H','I','J']
# model_root = './checkpoint/weights'
# for det_model in model_list:
# 	for dataset in dataset_list:
# 		image_ext = 'JPG'
# 		if dataset =='H':
# 			image_ext = 'jpg'
# 		save_dir = os.path.join('Result',str(today),det_model,dataset)
# 		folder = os.path.join(dataset_root,dataset)
# 		model_dir = os.path.join(model_root,det_model,dataset)
# 		os.system('python inference_image_journal.py --image_root {}  --out_dir {} --evaluate True --det_model {} --image_ext {} --model_dir {}'.format(folder,save_dir,det_model,image_ext,model_dir))

#habitat
# today = date.today()
# model_list = ['faster','yolonas','retinanet','retinanetknn','yolo']
# dataset = 'drone_collection_habitat'
# dataset_root = './example_images'
# all_folders_dataset = glob.glob(dataset_root+'/'+dataset+'/test/*/*/')
# csv_root = os.path.join(dataset_root,dataset,'image_info.csv')
# for det_model in model_list:
# 	for folder in all_folders_dataset:
# 		height = folder.split(os.sep)[-3]
# 		subset = folder.split(os.sep)[-2]
# 		save_dir = os.path.join('Result',str(today),det_model,subset,height)
# 		os.system('python inference_image_height.py --image_root {}  --out_dir {} --evaluate True --det_model {} --image_ext {} --csv_root {}'.format(folder,save_dir,det_model,'JPG',csv_root))


#height
today = date.today()
model_list = ['faster','yolonas','retinanet','retinanetknn','yolo']
dataset = 'balanced_test_height'
dataset_root = 'example_images'
all_folders_dataset = glob.glob(os.path.join(dataset_root,dataset,'test','*'))
csv_root = os.path.join(dataset_root,dataset,'image_info.csv')
for det_model in model_list:
	for folder in all_folders_dataset:
		height = folder.split(os.sep)[-1]
		subset = folder.split(os.sep)[-3]
		save_dir = os.path.join('Result',str(today),subset,det_model,height)
		os.system('python inference_image_height.py --image_root {}  --out_dir {} --evaluate True --det_model {} --image_ext {} --csv_root {}'.format(folder,save_dir,det_model,'JPG',csv_root))