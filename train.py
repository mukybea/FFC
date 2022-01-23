import os
import torch 
import numpy as np
from tqdm import tqdm



# print(device)



def trainer(epoch, train_loader_A, train_loader_B, device, model, optimizer, criterion):
    

    idx = 0
    # for epoch in range(0,10):
    total_correct = 0
    no_of_samples_so_far = 0
    loss_so_far = 0
    with tqdm(train_loader_A, unit="batch") as tepoch:
     for (view_a,label), (view_b,_) in zip(tepoch, train_loader_B):
        batch_size = view_a.size(0)
        # print(view_b.shape, view_a.shape)
        tepoch.set_description(f"Epoch {epoch+1}")
        views = torch.stack([view_a.to(device), view_b.to(device)]).transpose(1,0)
        # print(views.shape)
        # print("views a shape", views[0].shape)
        # print("views b shape", views[1].shape)
        # print("label shape", label.shape)
        # print(label[:10])
        optimizer.zero_grad()
        out = model(views)
        label = label.to(device)
        loss = criterion(out, label.view(-1))
        loss_so_far += loss.item() * batch_size
        loss.backward()
        optimizer.step()
        outputs_1 = out.argmax(dim=1, keepdim=True).squeeze()
        correct = (outputs_1 == label).sum().item()
        total_correct += correct
        no_of_samples_so_far += batch_size
        loss_out = loss_so_far/no_of_samples_so_far
        acc = total_correct/no_of_samples_so_far

        tepoch.set_postfix(loss=loss_out, accuracy=100. * acc)
        # if idx ==30:
        #   break;
        # idx += 1

    torch.save(model.state_dict(), os.path.join(os.getcwd(),"saved_models/model.pth"))
        #  print("final_accuracy: ", total_correct/len(train_loader_A.dataset)