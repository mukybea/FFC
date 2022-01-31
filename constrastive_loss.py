import torch 
import torch.nn as nn
import torch.nn.functional as F
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        # self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        # self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
    
    def get_device(self):
        if torch.cuda.is_available(): 
            device = 'cuda:0' 
        else: 
            device = 'cpu'
        return device

    def global_pool(self, emb_a, emb_b):

        self.emb_a = emb_a.view(emb_a.size(0),emb_a.size(1), -1)
        self.emb_b = emb_b.view(emb_b.size(0), emb_b.size(1), -1)

        return self.emb_a.mean(dim=-1), self.emb_b.mean(dim=-1)

    def forward(self, emb_i, emb_j, batch_size2):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        # print(emb_i.shape, emb_j.shape)
        device = self.get_device()
        self.batch_size = batch_size2
        # self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size2 * 2, batch_size2 * 2, dtype=bool)).float())
           

        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        z_i, z_j = self.global_pool(z_i, z_j)
        # print(z_i.shape, z_j.shape)

        representations = torch.cat([z_i, z_j], dim=0)
        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        positives = positives.to(device)
        self.negatives_mask = self.negatives_mask.to(device)

        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss