'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self, train=True):
        """
        Load the dataset
        :param train: boolean, True for loading training data, False for loading testing data
        :return: a dictionary containing features and labels
        """
        print('loading data...')
        file_name = 'train.csv' if train else 'test.csv'
        X = []
        y = []
        f = open(self.dataset_source_folder_path + file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            y.append(elements[0])  # First element is the label
            X.append(elements[1:])  # Remaining elements are features
        f.close()
        print(f'Loaded {"training" if train else "testing"} data with {len(X)} instances')
        return {'X': X, 'y': y}