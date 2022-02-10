import os
import torch 
import numpy as np
from tqdm import tqdm



# print(device)



def trainer(epoch, train_loader, device, model, optimizer, criterion_1, criterion_2):
    

    idx = 0
    # for epoch in range(0,10):
    total_correct = 0
    no_of_samples_so_far = 0
    loss_contr = 0
    loss_so_far = 0
    with tqdm(train_loader, unit="batch") as tepoch:
     for (view_a, view_b, label) in tepoch:
        batch_size = view_a.size(0)
        # print(view_b.shape, view_a.shape)
        # print(torch.sum(view_a), torch.sum(view_b))
        # print("equal") if (torch.sum(view_a)==torch.sum(view_b)) else print("not equal")

        tepoch.set_description(f"Epoch {epoch+1}")
        # views = torch.stack([view_a.to(device), view_b.to(device), view_c.to(device)]).transpose(1,0)
        views = torch.stack([view_a.to(device), view_b.to(device)]).transpose(1,0)
        # print(views.shape)
        # print("views a shape", views[0].shape)
        # print("views b shape", views[1].shape)
        # print("label shape", label.shape)
        # print("label view(-1)", label.view(-1), label.view(-1).shape)
        # print("label only", label)
        # print(label[:10])
        optimizer.zero_grad()
        out, emb1, emb2 = model(views)
        # print("out only ",out)
        # print("out indv", out[0])
        label = label.to(device)
        loss_1 = criterion_1(out, label.view(-1))
        loss_2 = criterion_2(emb1, emb2, batch_size)
        loss = (5*loss_1) + (3*loss_2)
        # loss = loss_1
        # loss = 5*(criterion_1(out, label.view(-1))) + (3.0 * contrastive_loss)
        loss_so_far += loss_1.item() * batch_size
        loss_contr += loss_2.item() * batch_size
        loss.backward()
        optimizer.step()
        outputs_1 = out.argmax(dim=1, keepdim=True).squeeze()
        correct = (outputs_1 == label).sum().item()
        total_correct += correct
        no_of_samples_so_far += batch_size
        loss_out = loss_so_far/no_of_samples_so_far
        loss_contr_out = loss_contr / no_of_samples_so_far
        acc = total_correct/no_of_samples_so_far

        tepoch.set_postfix(loss=loss_out, accuracy=100. * acc, contrastive_loss=loss_contr_out)
        # tepoch.set_postfix(loss=loss_out, accuracy=100. * acc, contrastive_loss=contrastive_loss.item())
        # if idx ==30:
        #   break;
        # idx += 1

    torch.save(model.state_dict(), os.path.join(os.getcwd(),"saved_models/model.pth"))
        #  print("final_accuracy: ", total_correct/len(train_loader_A.dataset)