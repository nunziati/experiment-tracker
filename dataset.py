import torch
import torchvision
from torchvision import transforms
import random

class ETDataset(torch.utils.data.Dataset):
    """Class that implements a dataset that is able to merge more datasets in one in a transparent way."""
    
    def __init__(self, datasets_list, set_labels_from=None, input_preprocessing_function=None, output_preprocessing_function=None, tasks=None):
        """Initialize the dataset.
        Args:
            datasets_list: an iterable of datasets: the datasets that we want to merge.
            set_labels_from: an instance of this class, that is used to create the labels in a consistent way; if None, the labels are set automatically.
            dataloader_args: dictionary of arguments to be given to the dataloader (if None, no dataloader is used)
        """

        if isinstance(datasets_list, list) or isinstance(datasets_list, tuple):
            self.datasets_list = list(datasets_list)
        else:
            self.datasets_list = [datasets_list]

        self.class_map = {} # to map the label unique identifier with the couple (dataset_id, label_within_the_dataset)
        self.reverse_class_map = {}
        data_indexes_list = []

        # copying the labels from another dataset (e.g. useful when creating a test set, and want to keep the same labels of a training set)
        if set_labels_from is not None:
            self.class_map = set_labels_from.class_map

        # building a 2D tensor with columns (dataset_id, example_id_whithin_the_dataset)
        for dataset_index, dataset in enumerate(self.datasets_list):
            n_samples = len(dataset)

            # creating the two columns for the considered dataset
            dataset_index_array = torch.full((1, n_samples), dataset_index, dtype=torch.int64)
            example_index_array = torch.arange(n_samples, dtype=torch.int64).reshape((1, -1))

            # concatenating the two columns
            data_index_array = torch.cat((dataset_index_array, example_index_array))
            
            # preparing the columns to be concatenated with other columns
            data_indexes_list.append(data_index_array)
            
            # extract the labels of each dataset and give them a unique id
            if set_labels_from is None:
                self.class_map.update({
                    (dataset_index, x.item()): x.item() + len(self.class_map)
                    for x in torch.unique(torch.LongTensor(dataset.targets), sorted=True)
                })
        
        # concatenating the tables of each dataset
        self.data_indexes = torch.cat(data_indexes_list, dim=1).transpose(0, 1)

        self.reverse_class_map = {self.class_map[x]: x for x in self.class_map}
        self.num_classes = len(self.class_map)

        self.input_preprocessing_function = (
            input_preprocessing_function if input_preprocessing_function != None else lambda x: x
        )
        self.output_preprocessing_function = (
            output_preprocessing_function if output_preprocessing_function != None else lambda x: x
        )

        self.tasks = tasks if tasks is not None else [(x,) for x in self.reverse_class_map]

    def get_raw_item(self, index):
        """Get an example from the dataset, without applying the preprocessing function.
        
        Args:
            index: can be an int or a couple (dataset_id, example_id_whithin_the_dataset).
        Returns:
            The pair (pattern, label)
        """

        if isinstance(index, int):
            index = self.data_indexes[index]

        dataset_index = index[0].item()
        example_index = index[1]
        data, label = self.datasets_list[dataset_index][example_index]
        label = self.class_map[(dataset_index, label)] # uses the unique identifier of the label
        return data, label

    def preprocess(self, data, label):
        processed_data = self.input_preprocessing_function(data)
        processed_label = self.output_preprocessing_function(label)

        return processed_data, processed_label

    def __getitem__(self, index):
        """Get an example from the dataset, applying the preprocessing function.
        
        Args:
            index: can be an int or a couple (dataset_id, example_id_whithin_the_dataset).
        Returns:
            The pair (pattern, label)
        """
        return self.preprocess(*self.get_raw_item(index))

    def __len__(self):
        """Return the total number of examples in this dataset."""

        return self.data_indexes.shape[0]

    def shuffle(self, groups=None):
        """Shuffle the examples of the dataset, mixing togeher examples of different datasets."""

        if groups == None:
            indexes = torch.randperm(self.data_indexes.shape[0])
            self.data_indexes = self.data_indexes[indexes]
        else:
            # Idea: shuffle only within groups of labels
            raise NotImplementedError

    def dataset_wise_sort_by_label(self):
        """Sort the dataset by target class id."""

        self.shuffle()

        data_indexes = []

        for task in self.tasks:
            data_indexes.extend(
                [idx for idx in self.data_indexes if self.get_raw_item(idx)[1] in task]
            )

        return data_indexes

    def get_subset_by_label(self, label):
        indices = [idx for idx in range(len(self)) if self[idx][1] == label]
        
        return torch.utils.data.Subset(self, indices)

    def split(self, split_proportions):
        assert abs(sum(split_proportions) - 1) <= 1e-5, "The splitting proportions should sum to 1."

        split_proportions = [int(x * len(self)) for x in split_proportions]
        indices = list(range(len(self)))
        random.shuffle(indices)
        split_indices = []
        print(split_proportions)

        for num_examples in split_proportions:
            split_indices.append(torch.utils.data.Subset(self, indices[-num_examples:]))
            del indices[-num_examples:]

        return tuple(split_indices)

def get_dataset(torchvision_dataset, normalization=((), ()), tasks=None, root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalization)
    ])

    training_set = torchvision_dataset(root=root, train=True, download=True, transform=transform)
    test_set = torchvision_dataset(root=root, train=False, download=True, transform=transform)

    return ETDataset(training_set), ETDataset(test_set)


def get_MNIST_dataset(tasks=None, root='./data'):
    return get_dataset(
        torchvision.datasets.MNIST,
        normalization=((0.1307,), (0.3081,)),
        tasks=tasks,
        root=root
    )


def get_CIFAR10_dataset(tasks=None, root='./data'):
    return get_dataset(
        torchvision.datasets.CIFAR10,
        normalization=((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        tasks=tasks,
        root=root
    )


def get_CIFAR100_dataset(tasks=None, root='./data'):
    return get_dataset(
        torchvision.datasets.CIFAR100,
        normalization=((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025)),
        tasks=tasks,
        root=root
    )


def get_splitMNIST_dataset(root='./data'):
    return get_MNIST_dataset(tasks=[(2 * x, 2 * x + 1) for x in range(5)], root=root)

def get_splitCIFAR100_dataset(root='./data'):
    return get_CIFAR100_dataset(tasks=[tuple([5 * x + i for i in range(5)]) for x in range(20)], root=root)