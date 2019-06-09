import numpy as np
import os
import pickle
import torch

from torch.utils.data import Dataset, DataLoader
from scipy.misc import imresize

CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5
}

# class RawDataset(Dataset):
#     def __init__(self, directory, dataset="train"):
#         self.instances, self.labels = self.read_dataset(directory, dataset)

#         self.instances = torch.from_numpy(self.instances)
#         self.labels = torch.from_numpy(self.labels)

#     def __len__(self):
#         return self.instances.shape[0]

#     def __getitem__(self, idx):
#         sample = { 
#             "instance": self.instances[idx],
#             "label": self.labels[idx] 
#         }

#         return sample

#     def zero_center(self, mean):
#         self.instances -= float(mean)

#     def read_dataset(self, directory, dataset="train"):
#         if dataset == "train":
#             filepath = os.path.join(directory, "train.p")
#         elif dataset == "dev":
#             filepath = os.path.join(directory, "dev.p")
#         else:
#             filepath = os.path.join(directory, "test.p")

#         videos = pickle.load(open(filepath, "rb"))

#         instances = []
#         labels = []
#         for video in videos:
#             for frame in video["frames"]:
#                 instances.append(frame.reshape((1, 64, 64)))
#                 labels.append(CATEGORY_INDEX[video["category"]])

#         instances = np.array(instances, dtype=np.float32)
#         labels = np.array(labels, dtype=np.uint8)

#         self.mean = np.mean(instances)

#         return instances, labels

class BlockFrameDataset(Dataset):
    def __init__(self, directory, dataset="train", shape=None):
        self.shape = shape
        self.instances, self.labels = self.read_dataset(directory, dataset)

        self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)
        

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        sample = { 
            "instance": self.instances[idx], 
            "label": self.labels[idx] 
        }

        return sample

    def zero_center(self, mean):
        self.instances -= float(mean)

    def normalize(self):
        self.instances = self.instances / float(255)

    def read_dataset(self, directory, dataset="train", mean=None):
        desired_categories = [0,1,2,3,4, 5]

        if dataset == "train":
            filepath = os.path.join(directory, "train.p")
        elif dataset == "dev":
            filepath = os.path.join(directory, "dev.p")
        else:
            filepath = os.path.join(directory, "test.p")

        videos = pickle.load(open(filepath, "rb"))

        instances = []
        labels = []
        current_block = []
        for video in videos:
            if CATEGORY_INDEX[video["category"]] in desired_categories:
                for i, frame in enumerate(video["frames"]):
                    if self.shape:
                        frame = imresize(frame, size=(self.shape, self.shape))
                    current_block.append(frame)
                    if len(current_block) % 15 == 0:
                        current_block = np.array(current_block)
                        if self.shape:
                            instances.append(current_block.reshape((1, 15, self.shape, self.shape)))
                        else:
                            instances.append(current_block.reshape((1, 15, 128, 128)))
                        labels.append(CATEGORY_INDEX[video["category"]])
                        current_block = []

        instances = np.array(instances, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        self.mean = np.mean(instances)
        
        return instances, labels

     
class MovingMNIST(Dataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set. 
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    urls = [
        'https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(self, root, train=True, split=1000, transform=None, target_transform=None, download=False, shape=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train  # training set or test set
        self.shape = shape


        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if train:
            self.train_data = self.read_dataset()
        else:
            self.test_data = self.read_dataset()

    def __getitem__(self, idx):

        if self.train:
            sample = { 
                "instance": self.train_data[idx], 
                "label": "" 
            }
        else:
            sample = { 
                "instance":  self.test_data[idx], 
                "label": "" 
            }

        return sample

    def zero_center(self, mean):
        self.instances -= float(mean)

    def normalize(self):
        self.instances = self.instances / float(255)

    def read_dataset(self, mean=None):

        full_set = np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)#[:-self.split]

        instances = []
        for video in full_set:
            current_block = []
            for i, frame in enumerate(video):
                if self.shape:
                    frame = imresize(frame, size=(self.shape, self.shape))
                current_block.append(frame)

            current_block = np.array(current_block)
            if self.shape:
                instances.append(current_block.reshape((1, 20, self.shape, self.shape)))
            else:
                instances.append(current_block.reshape((1, 20, 128, 128)))


        instances = np.array(instances, dtype=np.float32)

        #train = instances[:-self.split]
        #test = instances[-self.split:]

        self.instances = instances[:-self.split] if self.train else instances[-self.split:]
        self.mean = np.mean(self.instances)
        
        return torch.from_numpy(self.instances)

    #def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """
        #if self.train:
        #    seq, target = self.train_data[index, :], self.train_data[index, :]
        #else:
        #    seq, target = self.test_data[index, :], self.test_data[index, :]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # seq = Image.fromarray(seq.numpy(), mode='L')
        # target = Image.fromarray(target.numpy(), mode='L')

        # if self.transform is not None:
        #     seq = self.transform(seq)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        #return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_train_seq.npy')).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str