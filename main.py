import os
import torch
import torch.nn as nn
from models import Builds
from dataset import train_test_loader 
import train
import test
import metrics
import constrastive_loss 
from torch.optim.lr_scheduler import StepLR
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


	# def count_parameters(model):
	# 	    a = sum(p.numel() for p in model.parameters() if p.requires_grad)
	# 	    b = sum(p.numel() for p in model.parameters())
	# 	    return a,b


	device = get_device()
	model = Builds().to(device)
	# a,b = count_parameters(model)
	# print(a,b)
	# for o, p in model.named_parameters():
	# 	print("named parameters ", o)

	# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.1)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5, 0.999))
	criterion_1 = nn.CrossEntropyLoss()
	criterion_2 = constrastive_loss.ContrastiveLoss(temperature=0.5)

	load_model = Builds()
	
	train_loader = train_test_loader.train_loader()
	test_loader = train_test_loader.test_loader()

	# scheduler = StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1, verbose=False)


	for epoch in range(0,100):
		# break

		train.trainer(epoch, train_loader, device, model, optimizer, criterion_1, criterion_2)
		load_model.load_state_dict(torch.load(os.path.join(os.getcwd(),"saved_models/model.pth")))
		load_model.to(device)
		track_label, track_output = test.tester(epoch, test_loader, device, load_model, criterion_2)
		# scheduler.step()

	precision, recall, f1_score = metrics.cal_metric(track_label, track_output)
	print("precision: ", precision)
	print("recall: ", recall)
	print("f1-score: ", f1_score)

	metrics.plot_conf_matrix(track_label, track_output)


if __name__ == '__main__':
	main()