import os
import torch
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

transform=torchvision.transforms.ToTensor()


class MultiviewDataset(Dataset):
    def __init__(self, first_view_path, second_view_path, transform=None):
        self.first_view_path = first_view_path
        self.second_view_path = second_view_path
        self.categories = {cat: i for i, cat in enumerate(sorted(os.listdir(self.first_view_path)))}
        self.transform = transform
        self.first_view_img_paths = {}
        self.second_view_img_paths = {}
        self.index_to_class = {}
        for c in self.categories.keys():
            first_view_files = [x for x in sorted(os.listdir(os.path.join(self.first_view_path, c)))
                                if x.endswith('.jpg')
                                or x.endswith('.jpeg')
                                or x.endswith('.png')
                                or x.endswith('.tif')]
            second_view_files = [x for x in sorted(os.listdir(os.path.join(self.second_view_path, c)))
                                 if x.endswith('.jpg')
                                 or x.endswith('.jpeg')
                                 or x.endswith('.png')
                                 or x.endswith('.tif')]

            for fname in first_view_files:
                basename = fname.split('.')[0]
                self.first_view_img_paths[c + '_' + basename] = os.path.join(first_view_path, c, fname)
                self.index_to_class[c + '_' + basename] = c
            for fname in second_view_files:
                basename = fname.split('.')[0]
                self.second_view_img_paths[c + '_' + basename] = os.path.join(second_view_path, c, fname)
                
        first_view_keys = sorted(list(self.first_view_img_paths.keys()))
        second_view_keys = sorted(list(self.second_view_img_paths.keys()))
        assert first_view_keys == second_view_keys, print(first_view_keys, second_view_keys)
        self.keys = first_view_keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        first_img_path = self.first_view_img_paths[key]
        second_img_path = self.second_view_img_paths[key]
        first_view = Image.open(first_img_path).convert('RGB')
        # print(first_view.size)
        second_view = Image.open(second_img_path).convert('RGB')
        # print(second_view.size)
        label = self.index_to_class[key]
        label = self.categories[label]
        if self.transform is not None:
            first_view = self.transform(first_view)
            second_view = self.transform(second_view)
        # print(first_view.shape, second_view.shape)
        return first_view, second_view, label


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
        multiview_dataset = MultiviewDataset('/storage/research/mview/train/aerial',
                                             '/storage/research/mview/train/ground',
                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Resize((500,500))]))
        
        train_loader = torch.utils.data.DataLoader(multiview_dataset,
                                              batch_size=64,
                                              shuffle=True)


        # view_A_trainset = torchvision.datasets.ImageFolder(root='../../../storage/research/mview/train/aerial', transform=transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((500,500))
        #     ]))

        # view_B_trainset = torchvision.datasets.ImageFolder(root='../../../storage/research/mview/train/ground', transform=transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((500,500))
        #     ]))

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

        # train_loader_A = torch.utils.data.DataLoader(view_A_trainset,
        #                                           batch_size=64,
        #                                           shuffle=False
        #                                           #  num_workers=2
        #                                           )
        # train_loader_B = torch.utils.data.DataLoader(view_B_trainset,
        #                                           batch_size=64,
        #                                           shuffle=False
        #                                           #  num_workers=2
        #                                           )

        return train_loader

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
        multiview_test_dataset = MultiviewDataset('/storage/research/mview/test/aerial',
                                                  '/storage/research/mview/test/ground',
                                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                                transforms.Resize((500,500))]))
        test_loader = torch.utils.data.DataLoader(multiview_test_dataset,
                                             batch_size=64,
                                             shuffle=True)

        # view_A_testset = torchvision.datasets.ImageFolder(root='../../../storage/research/mview/test/aerial', transform=transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((500,500))]))

        # view_B_testset = torchvision.datasets.ImageFolder(root='../../../storage/research/mview/test/ground', transform=transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((500,500))]))

        # test_loader_A = torch.utils.data.DataLoader(view_A_testset,
        #                                           batch_size=64,
        #                                           shuffle=False
        #                                           # num_workers=2
        #                                          )

        
        # test_loader_B = torch.utils.data.DataLoader(view_B_testset,
        #                                           batch_size=64,
        #                                           shuffle=False
        #                                           # num_workers=2
        #                                          )

        return test_loader
