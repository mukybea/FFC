import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Tanh


class Attention(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1x = nn.Sequential(
        nn.Conv2d(64, 64, 1, 1, padding="same"),
        nn.ReLU()
    )
  def forward(self,x):
      xc = self.conv1x(x)
      # print("xxc",xc.shape)
      return F.softmax(self.conv1x(x), dim=3)

class Conv_BN_RELU(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1xx = nn.Sequential(
        nn.Conv2d(32,32,1),
        nn.BatchNorm2d(32),
        nn.ReLU()
    )

  def forward(self, x):
      return self.conv1xx(x)

class CNNenc(nn.Module):
  def __init__(self):
      super().__init__()

      self.conv1 = nn.Sequential(
      nn.Conv2d(3, 16, 3, 3),
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.Conv2d(16, 32, 3, 2),
      nn.ReLU(),
      nn.Conv2d(32, 64, 3, 1),
      nn.BatchNorm2d(64),
      nn.Sigmoid()
      )

      self.conv2 = nn.Sequential(
      nn.Conv2d(3, 16, 3, 3),
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.Conv2d(16, 32, 3, 2),
      nn.ReLU(),
      nn.Conv2d(32, 64, 3, 1),
      nn.BatchNorm2d(64),
      nn.Sigmoid()
      )
      
      self.attend1 = Attention()
      self.attend2 = Attention()
      

  def forward(self, x):
    batch_size = x.size(1)
    # print("batch", batch_size)
    # x1 = x[0]
    # x2 = x[1]
    x1 = x.transpose(1,0)[0]
    x2 = x.transpose(1,0)[1]
    # print("x1 shape",x1.shape)
    # print("x2 shape",x2.shape)

    enc1 = self.conv1(x1)
    enc2 = self.conv2(x2)
    # print(enc1.shape)
    att_enc1 = self.attend1(enc1)
    att_enc2 = self.attend2(enc2)

    post_att1 = torch.matmul(enc1, att_enc1) #element wise multiplication
    post_att2 = torch.matmul(enc2, att_enc2)

    return post_att1, post_att2

class FFC(nn.Module):
  def __init__(self):
    super().__init__()
    self.split_lg = nn.Conv2d(64, 32, 1)
    self.l_l = nn.Conv2d(32,32,3,padding="same")
    self.l_g = nn.Conv2d(32,32,3,padding="same")
    self.g_l = nn.Conv2d(32,32,3,padding="same")
    self.conv1x1 = nn.Conv2d(32, 32, 1)
    self.bn_relu = nn.Sequential(
    nn.BatchNorm2d(32),
    nn.ReLU()
    )
    self.bn_relu_2 = nn.Sequential(
    nn.BatchNorm2d(32),
    nn.ReLU()
    )

    self.conv1x = Conv_BN_RELU()
    self.conv2x = nn.Sequential(
        nn.Conv2d(64,32,1)
    )

  def forward(self,x_1, x_2, batch_size):

    if x_2 is not None:
      # print("x2 shape is ", x_2.shape)
      xsplit1_a = self.split_lg(x_1)
      xsplit2_a = torch.clone(xsplit1_a)
      xsplit1_b = self.split_lg(x_2)
      xsplit2_b = torch.clone(xsplit1_b)

      xsplit1 = torch.add(xsplit1_a, xsplit1_b)
      xsplit2 = torch.add(xsplit2_a, xsplit2_b)
      # print("split_lg", xsplit2.shape, xsplit1.shape)
    else:
      xsplit1 = self.split_lg(x_1)
      xsplit2 = torch.clone(xsplit1)

    xl_l = self.l_l(xsplit1)
    xl_g = self.l_g(xsplit1)
    xg_l = self.g_l(xsplit2)
    # print("xg_l", xg_l.shape)
    # print("xl_g", xl_g.shape)
    #Spectral Transform block
    conv_bn_relu_1 =  self.conv1x(xsplit2)
    # print("conv_bn_relu_1", conv_bn_relu_1.shape)
    fft_dim = (-3, -2, -1) 
    fft = torch.fft.rfftn(conv_bn_relu_1, dim=fft_dim, norm='ortho')
    # fft = torch.fft.rfftn(conv_bn_relu_1)
    # print("fft ",fft.real.shape, fft.imag.shape)
    ffted = torch.stack((fft.real, fft.imag), dim=-1)
    ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
    ffted = ffted.view((batch_size, -1,) + ffted.size()[3:])
    # print("ffted", ffted.shape)
    

    conv_bn_relu_2 = self.bn_relu_2(self.conv2x(ffted))
    # print("convt", conv_bn_relu_2.shape)
    ifft = torch.fft.irfftn(conv_bn_relu_2)
    # print(ifft.shape)
    spectral_out = torch.add(ifft, conv_bn_relu_1)
    spectral_out_2 = self.conv1x1(spectral_out)
    # print("spectral_out_2", xl_g.shape, spectral_out_2.shape)

    local_feat = self.bn_relu(torch.add(xl_g, spectral_out_2))
    global_feat = self.bn_relu(torch.add(xl_l, xg_l))
    # print("local_feat --", local_feat.shape, global_feat.shape)

    feat_out = torch.cat((local_feat, global_feat), dim=1)
    
    # print("FT",feat_out.shape)
    # feat_out = feat_out.view(feat_out.size(0), feat_out.size(1), -1)
    # feat_out = feat_out.mean(dim=-1)
    # print(feat_out.shape)



    return feat_out

class Neural_classifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.classify = nn.Sequential(
        # nn.Linear(15618304, 1024),
        # nn.BatchNorm1d(64),
        # nn.Sigmoid(),
        nn.Linear(64, 11),
        nn.Tanh()
    )
    # self.lin1 = nn.Linear(x_size, 1024)
    # self.lin2 = nn.Linear(1024, nc)
    # self.tanh = nn.Tanh()
    # self.sig = nn.Sigmoid()

  # def get_device():
  #     if torch.cuda.is_available():
  #         device = 'cuda:0'
  #     else:
  #         device = 'cpu'
  #     return device

  def forward(self, x):
      # self.device = get_device()
      x_size = x.size(1)
      # print(x.size())
      x = self.classify(x)
      # x = F.linear(x, torch.randn(1024, x_size).to(self.device))
      # x = F.linear(x, torch.randn(1024, x_size))
      # x = self.lin1(x)
      # x = self.sig(x)
      # x = F.linear(x, torch.randn(nc, x.size(1)))
      # x = F.linear(x, torch.randn(nc, x.size(1)).to(self.device))
      # x = self.lin2(x)
      # x = self.tanh(x)
      return x

class Builds(nn.Module):
  def __init__(self):
    # def get_device():
    #   if torch.cuda.is_available():
    #       device = 'cuda:0'
    #   else:
    #       device = 'cpu'
    #   return device

    super().__init__()

    # self.device = get_device()
    # self.convenc = CNNenc().to(self.device)
    # self.ffc = FFC().to(self.device)
    # self.classfy = Neural_classifier().to(self.device)
    self.convenc = CNNenc()
    self.ffc = FFC()
    self.classfy = Neural_classifier()
    # self.n_class = n_class

  def forward(self, x):
    batch_size = x.size(0)
    # n_class = self.n_class

    xenc_1, xenc_2 = self.convenc(x)
    # print("enc 1", xenc_1.shape)
    xffced1 = self.ffc(xenc_1, xenc_2, batch_size)
    xffced2 = self.ffc(xffced1, None, batch_size)

    xcat = torch.add(xenc_1, xffced2)
    # print("FT",feat_out.shape)
    feat_out = xcat.view(xcat.size(0), xcat.size(1), -1)
    feat_out = feat_out.mean(dim=-1)
    # print("after global pool",feat_out.shape)

    # xcat = xcat.reshape(batch_size,-1)
    # print("--- xcat ----- ", xcat.size())

    classified = self.classfy(feat_out)

    return classified