import os
import torch
import torch.nn as nn
from models import Builds
from dataset import train_test_loader 
import train
import test
import metrics

def main():

	if os.path.exists(os.path.join(os.getcwd(), 'saved_models')):
		pass
	else:
		os.mkdir(os.path.join(os.getcwd(), 'saved_models'))
		
	def get_device():
	    if torch.cuda.is_available(): 
	    	device = 'cuda:0' 
	    else: 
	    	device = 'cpu'
	    return device

	device = get_device()
	model = Builds().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.5, 0.999))
	criterion = nn.CrossEntropyLoss()

	load_model = Builds()
	
	train_loader = train_test_loader.train_loader()
	test_loader = train_test_loader.test_loader()


	# print("len of train_loader_A", len(train_loader_A.dataset))
	# print("len of train_loader_B", len(train_loader_B.dataset))
	# for idxx, (im, lb,lbl) in enumerate(train_loader):
	# 	print(im.shape, lb.shape, lbl)
	# 	# print(lb[8])
	# 	if idxx ==20:
	# 			break
	# xc = os.listdir(os.path.join(os.getcwd(), "datasets/Train/view_1/View1_photo"))
	# labels = 
	# print(test_loader_A.dataset.class_to_idx)
	# for  idx, (ad, af) in enumerate((test_loader_B)):
	# 	if idx%10 == 0:
	# 		print(af[:10])
	# 	# print(ad.shape)
	# 	# if idxx == 3:
	# 	# 	break
	# 		# break
	# 	elif idx == 500:
	# 		break

	for epoch in range(0,50):
		# break

		train.trainer(epoch, train_loader, device, model, optimizer, criterion)
		load_model.load_state_dict(torch.load(os.path.join(os.getcwd(),"saved_models/model.pth")))
		load_model.to(device)
		track_label, track_output = test.tester(epoch, test_loader, device, load_model)

	precision, recall, f1_score = metrics.cal_metric(track_label, track_output)
	print("precision: ", precision)
	print("recall: ", recall)
	print("f1-score: ", f1_score)

	metrics.plot_conf_matrix(track_label, track_output)


if __name__ == '__main__':
	main()