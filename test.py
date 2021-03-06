import torch 
from tqdm import tqdm


def tester(epoch, test_loader, device, load_model,criterion_2):

    track_output = torch.zeros(64).to(device)
    track_label = torch.zeros(64).to(device)
    # epoch = 0
    idx = 0
    total_correct = 0
    no_of_samples_so_far = 0
    # trch = torch.zeros_like()
    with torch.no_grad():
      with tqdm(test_loader, unit='B') as tepoch:
        for view_a, view_b, label in test_loader:
          batch_size = view_a.size(0)
          tepoch.set_description(f"Test epoch {epoch+1}")
          views = torch.stack([view_a.to(device), view_b.to(device)]).transpose(1,0)
          # view_a = view_a.to(device)
          out, _, _ = load_model(views)
          label = label.to(device)
          outputs_1 = out.argmax(dim=1, keepdim=True).squeeze()
          correct = (outputs_1 == label).sum().item()
          total_correct += correct
          no_of_samples_so_far += batch_size
          acc = total_correct/no_of_samples_so_far

          tepoch.set_postfix(accuracy=100. * acc)

          if idx == 0:
            track_output = torch.add(track_output, outputs_1)
            track_label = torch.add(track_label, label)
          else:
            torch.cat((track_output, outputs_1), 0)
            torch.cat((track_label, label), 0)
          
          # if (idx >= 100):
          #   break;
          idx += 1

    return track_label, track_output
