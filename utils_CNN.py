import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import copy
import urllib.request
import pickle
from PIL import Image

def get_task_load_train(train_dataset,batch_size):
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size,
    num_workers=8,
    pin_memory=True, shuffle=True)
    print('Train loader length', len(train_loader))    
    return train_loader
 
def get_task_load_test(test_dataset,test_batch_size):
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    return test_loader

def load_MNIST():
    MNIST_normalize = transforms.Normalize((0.1, 0.1, 0.1), (0.2752, 0.2752, 0.2752))
    MNIST_train_transform = transforms.Compose([
        transforms.Pad(padding=2, fill=0),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        MNIST_normalize
        ])

    MNIST_test_transform = transforms.Compose([
        transforms.Pad(padding=2, fill=0),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        MNIST_normalize
    ])
    MNIST_train = datasets.MNIST('_data', train=True, download=True, transform=MNIST_train_transform)
    MNIST_test = datasets.MNIST('_data', train=False, transform=MNIST_test_transform)

    return MNIST_train, MNIST_test

def load_FashionMNIST():
    FashionMNIST_normalize = transforms.Normalize((0.2190, 0.2190, 0.2190), (0.3318, 0.3318, 0.3318))
    FashionMNIST_train_transform = transforms.Compose([
        transforms.Pad(padding=2, fill=0),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        FashionMNIST_normalize
        ])

    FashionMNIST_test_transform = transforms.Compose([
        transforms.Pad(padding=2, fill=0),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        FashionMNIST_normalize
    ])
    FashionMNIST_train = datasets.FashionMNIST('_data', train=True, download=True, transform=FashionMNIST_train_transform)
    FashionMNIST_test = datasets.FashionMNIST('_data', train=False, transform=FashionMNIST_test_transform)

    return FashionMNIST_train, FashionMNIST_test

def load_notMNIST():
    normalize = transforms.Normalize((0.4254, 0.4254, 0.4254),
                                    (0.4501, 0.4501, 0.4501))

    NOTMNIST_train_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
        ])

    NOTMNIST_test_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
    ])
        
    train = notMNIST('_data', train=True, download=True, transform=NOTMNIST_train_transform)
    test = notMNIST('_data', train=False, download=True, transform=NOTMNIST_test_transform)
    return train, test

def load_cifar10():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                    (4,4,4,4),mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
         normalize
    ])

    full_dataset_cifar10 = datasets.CIFAR10('_data', train=True, transform=train_transform, download=True)
    test_dataset_cifar10 = datasets.CIFAR10('_data', train=False, transform=test_transform, download=False)

    return full_dataset_cifar10, test_dataset_cifar10

def load_cifar100():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                    (4,4,4,4),mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
         normalize
    ])
    full_dataset_cifar100 = datasets.CIFAR100('_data', train=True, transform=train_transform, download=True)
    test_dataset_cifar100 = datasets.CIFAR100('_data', train=False, transform=test_transform, download=False)
    return full_dataset_cifar100, test_dataset_cifar100


def task_construction(task_labels, target_task_labels, benchmark):
    if benchmark=='CIFAR10':
        train_dataset_cifar10, test_dataset_cifar10 = load_cifar10()
        train_dataset = split_dataset_by_labels(train_dataset_cifar10, task_labels, target_task_labels)
        test_dataset = split_dataset_by_labels(test_dataset_cifar10, task_labels, target_task_labels)
    elif benchmark=='CIFAR100':
        train_dataset_cifar100, test_dataset_cifar100 = load_cifar100()
        train_dataset = split_dataset_by_labels(train_dataset_cifar100, task_labels, target_task_labels)
        test_dataset = split_dataset_by_labels(test_dataset_cifar100, task_labels, target_task_labels)
    elif benchmark == 'mix':
        cifar_10_train, cifar_10_test = load_cifar10()
        MNIST_train, MNIST_test = load_MNIST()
        FashionMNIST_train, FashionMNIST_test = load_FashionMNIST()
        notMNIST_train, notMNIST_test = load_notMNIST()
        train_dataset = split_mix(cifar_10_train,
                    MNIST_train, FashionMNIST_train,
                    notMNIST_train,
                    task_labels, target_task_labels)
        test_dataset = split_mix(cifar_10_test,
            MNIST_test, FashionMNIST_test,
            notMNIST_test,
            task_labels, target_task_labels)
    return train_dataset,test_dataset

def split_mix(cifar_10, MNIST, FashionMNIST, notMNIST, task_labels, target_task_labels):
    #  cifar10 - MNIST - not MNIST - Fashion -  cifar10 - MNIST - not MNIST - Fashion
    cifar10_task_idx = [0,4]
    MNIST_task_idx = [1,5]
    notMNIST_task_idx = [2,6]
    FashionMNIST_task_idx = [3,7]
    
    datasets = []
    task_id = 0
    for labels, target_labels in zip(task_labels, target_task_labels):
        if task_id in cifar10_task_idx:
            idx=np.in1d(cifar_10.targets, labels)
            splited_dataset=copy.deepcopy(cifar_10)
        elif task_id in MNIST_task_idx:
            idx=np.in1d(MNIST.targets, labels)
            splited_dataset=copy.deepcopy(MNIST)
        elif task_id in FashionMNIST_task_idx:
            idx=np.in1d(FashionMNIST.targets, labels)
            splited_dataset=copy.deepcopy(FashionMNIST)
        elif task_id in notMNIST_task_idx:
            idx=np.in1d(notMNIST.targets, labels)
            splited_dataset=copy.deepcopy(notMNIST)
        targets = change_labels(labels, target_labels, torch.LongTensor(splited_dataset.targets)[idx])
        splited_dataset.targets = targets
        splited_dataset.data = splited_dataset.data[idx]
        datasets.append(splited_dataset)
        task_id+=1
    return datasets


def split_dataset_by_labels(dataset, task_labels, target_task_labels):
    datasets = []
    for labels, target_labels in zip(task_labels, target_task_labels):
        idx = np.in1d(dataset.targets, labels)
        splited_dataset = copy.deepcopy(dataset)
        targets = change_labels(labels, target_labels, torch.LongTensor(splited_dataset.targets)[idx])
        splited_dataset.targets = targets
        splited_dataset.data = splited_dataset.data[idx]
        datasets.append(splited_dataset)
    return datasets

def change_labels(current_labels, target_labels, targets):
    new_targets=copy.deepcopy(targets)
    for i in range(len(current_labels)):
        new_targets[targets==current_labels[i]] = target_labels[i]
    return new_targets

class notMNIST(torch.utils.data.Dataset):
    """The notMNIST dataset is a image recognition dataset of font glypyhs for the letters A through J useful with simple neural networks. It is quite similar to the classic MNIST dataset of handwritten digits 0 through 9.
    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self, root, train=True,transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "notmnist.zip"
        self.url = "https://github.com/nkundiushuti/notmnist_convert/blob/master/notmnist.zip?raw=true"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()

        training_file = 'notmnist_train.pkl'
        testing_file = 'notmnist_test.pkl'
        if train:
            with open(os.path.join(root,training_file),'rb') as f:
                train = pickle.load(f)
            self.data = train['features'].astype(np.uint8)
            self.targets = train['labels'].astype(np.uint8)
        else:
            with open(os.path.join(root,testing_file),'rb') as f:
                test = pickle.load(f)

            self.data = test['features'].astype(np.uint8)
            self.targets = test['labels'].astype(np.uint8)


    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img[0])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


def alloc_fix_count_per_layer(args, num_conv, feature_maps):
    print("selec_prec_conv",args.alloc_prec_conv)
    print("selec_prec_fc",args.alloc_prec_fc)
    print("selec_prec_fc_last_layer",args.alloc_prec_fc_last_layer)
    print("freezed_prec_conv",args.freezed_prec_conv)
    print("freezed_prec_fc",args.freezed_prec_fc)
    print("freezed_prec_fc_last_layer",args.freezed_prec_fc_last_layer)

    num_freezedNodes_per_layer = []
    selected_nodes_count = []
    num_freezedNodes_per_layer.append(0) #input_channel
    selected_nodes_count.append(3)
    for i in range(1, len(feature_maps)-1):
        if i <= num_conv:
            selected_nodes_count.append(int(args.alloc_prec_conv*feature_maps[i]))
            num_freezedNodes_per_layer.append(int(args.freezed_prec_conv*selected_nodes_count[i]))
        elif i == len(feature_maps)-2:
            selected_nodes_count.append(int(args.alloc_prec_fc_last_layer*feature_maps[i]))
            num_freezedNodes_per_layer.append(int(args.freezed_prec_fc_last_layer*selected_nodes_count[i])) 
        else:
            selected_nodes_count.append(int(args.alloc_prec_fc*feature_maps[i]))
            num_freezedNodes_per_layer.append(int(args.freezed_prec_fc*selected_nodes_count[i])) 

    selected_nodes_count.append(args.num_classes_per_task)
    num_freezedNodes_per_layer.append(args.num_classes_per_task)
    return selected_nodes_count, num_freezedNodes_per_layer

def sFree_reuse_count_per_layer(args, num_conv, feature_maps, l_reuse, selected_nodes_count):
    print("subfree_prec_conv",args.subfree_prec_conv)
    print("reuse_prec_conv",args.reuse_prec_conv)
    print("subfree_prec_fc",args.subfree_prec_fc)
    print("reuse_prec_fc",args.reuse_prec_fc)

    sim_sfree_nodes = []
    sim_reused_from_previous = []
    sim_sfree_nodes.append(3) #input
    sim_reused_from_previous.append(0)
    for i in range(1, len(feature_maps)-1):
        if i < l_reuse:
            sim_sfree_nodes.append(0)
            sim_reused_from_previous.append(0)
        else:
            if i <= num_conv:
                sim_sfree_nodes.append(int(args.subfree_prec_conv*selected_nodes_count[i]))
                sim_reused_from_previous.append(int(args.reuse_prec_conv*selected_nodes_count[i]))
            else:
                sim_sfree_nodes.append(int(args.subfree_prec_fc *selected_nodes_count[i]))
                sim_reused_from_previous.append(int(args.reuse_prec_fc*selected_nodes_count[i]))
    sim_sfree_nodes.append(0)
    sim_reused_from_previous.append(0)
    return sim_sfree_nodes, sim_reused_from_previous

def construct_class_order(class_order, benchmark, num_tasks, num_classes_per_task):
    if class_order=='None':
        class_order = list(range(0, num_tasks*num_classes_per_task))
    elif benchmark=='mix':
        class_order = []
        for i in range(num_tasks):
            if i < (num_tasks/2):
                class_order = [*class_order, *list(range(0, num_classes_per_task))]
            else:
                class_order= [*class_order, *list(range(num_classes_per_task, 2*num_classes_per_task))]
    elif benchmark=='mix_dataset':
        class_order = []
        for i in range(num_tasks):
            class_order = [*class_order, *list(range(0, num_classes_per_task))]
    else:
        class_order = list(map(int, class_order.split(',')))

    return class_order