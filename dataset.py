import torch
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

transform=torchvision.transforms.ToTensor()

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class train_test_loader:
    def __init__(self):
        self.addgaus = AddGaussianNoise()


    def lbp_transform(x):
        radius = 2
        n_points = 8 * radius
        METHOD = 'uniform'
        imgUMat = np.float32(x)
        # gray = cv2.cvtColor(imgUMat, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(imgUMat, n_points, radius, METHOD)
        lbp = torch.from_numpy(lbp).float()
        return lbp

    def gabor_transform(x):
        x = cv2.cvtColor(np.float32(x), cv2.COLOR_BGR2GRAY)
        # num = 0
        # for i in range (5):
        kernel = cv2.getGaborKernel((10, 10), result_sigma[i],
        result_theta[i], result_lambdaa[i],result_gamma[i], result_psi[i], ktype=cv2.CV_32F)
        gabor_label = str(num)  


        fimg = cv2.filter2D(np.float32(x), cv2.CV_8UC3, kernel)
        img_resized = cv2.resize (fimg, (32,32))
        img_tensor = torch.from_numpy(img_resized).long()

        return img_tensor

########################################

    def train_loader():

        # addgaus = AddGaussianNoise()

        # mnist_trainset_A = torchvision.datasets.MNIST(
        #     root = './data',
        #     train=True,
        #     download = True,
        #     transform = transforms.Compose([
        #         # transforms.Lambda(lbp_transform),
        #         # transforms.ToPILImage(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,)),
        #         # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #         # AddGaussianNoise(0, 0.5)
        #     ])
        # )

        view_A_trainset = torchvision.datasets.ImageFolder(root='../../../storage/research/mview/train/aerial', transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((500,500))
            ]))

        view_B_trainset = torchvision.datasets.ImageFolder(root='../../../storage/research/mview/train/ground', transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((500,500))
            ]))

        # mnist_trainset_B = torchvision.datasets.MNIST(
        #     root = './data',
        #     train=True,
        #     download = True,
        #     transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         # transforms.Normalize((0.1307, 0.1357, 0.1307), (0.229, 0.224, 0.225)),
        #         # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #         transforms.Normalize((0.1307,), (0.3081,)),
        #         AddGaussianNoise(0,0.3)
        #         # addgaus(0, 0.3)                                 
        #     ])
        # )

        train_loader_A = torch.utils.data.DataLoader(view_A_trainset,
                                                  batch_size=64,
                                                  shuffle=False
                                                  #  num_workers=2
                                                  )
        train_loader_B = torch.utils.data.DataLoader(view_B_trainset,
                                                  batch_size=64,
                                                  shuffle=False
                                                  #  num_workers=2
                                                  )

        return train_loader_A, train_loader_B

    def test_loader():

        # addgaus = AddGaussianNoise()
        # mnist_testset_A = torchvision.datasets.MNIST(
        #     root = './data',
        #     train=False,
        #     download = True,
        #     transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         # transforms.Normalize((0.1307,), (0.3081,)),
        #         # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #         # AddGaussianNoise(0, 0.5)                                 
        #     ])
        # )

        # mnist_testset_B = torchvision.datasets.MNIST(
        #     root = './data',
        #     train=False,
        #     download = True,
        #     transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,)),
        #         # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #         AddGaussianNoise(0,0.3)
        #         # addgaus(0, 0.3)                                 
        #     ])
        # )

        view_A_testset = torchvision.datasets.ImageFolder(root='../../../storage/research/mview/test/aerial', transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((500,500))]))

        view_B_testset = torchvision.datasets.ImageFolder(root='../../../storage/research/mview/test/ground', transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((500,500))]))

        test_loader_A = torch.utils.data.DataLoader(view_A_testset,
                                                  batch_size=64,
                                                  shuffle=False
                                                  # num_workers=2
                                                 )

        
        test_loader_B = torch.utils.data.DataLoader(view_B_testset,
                                                  batch_size=64,
                                                  shuffle=False
                                                  # num_workers=2
                                                 )

        return test_loader_A, test_loader_B
