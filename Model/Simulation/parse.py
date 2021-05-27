import os
import pandas as pd
import numpy as np
import arff as arff_module

SUPPORTED_FORMAT = ['csv', 'arff']


class Parse:
    '''
    class to read dataset(preprocessed)
    '''

    @classmethod
    def read_ds(cls, path, target_index=-1):
        '''
        main method to read file - work as mediator and send to the corrector format method
        :param path: url path of dataset
        :param target_index: the index of the target in the ds - default:0
        :return: feature data(X), target data(y), labels of the target(classes)
        '''
        file_name = os.path.basename(path)
        format = file_name.split(".")[-1].lower()
        if format in SUPPORTED_FORMAT:
            try:
                return getattr(Parse, format)(path, target_index)
            except Exception as e:
                print(e)
                raise Exception(f'Could not parse file:{str(e)}')
        else:
            raise Exception("File Format Not Supported")

    @classmethod
    def csv(cls, path, target_index):
        '''
        method to parse csv dataset file
        :param path: url path of dataset
        :param target_index: the index of the target in the ds - default:0
        :return: feature data(X), target data(y), labels of the target(classes)
        '''
        dataset = pd.read_csv(path)
        dataset = dataset[(dataset != '?').all(1)].to_numpy()
        X, y = cls.split_dataset(dataset, target_index)
        return X, y, list(np.unique(y))

    @classmethod
    def arff(cls, path, target_index):
        '''
        method to parse csv dataset file
        :param path: url path of dataset
        :param target_index: the index of the target in the ds - default:0
        :return: feature data(X), target data(y), labels of the target(classes)
        '''
        dataset = arff_module.load(open(path))

        for tup in dataset['attributes']:
            if tup[0] == 'target':
                classes = [int(target) for target in tup[1]]
                break

        dataset = np.array(dataset['data'])
        X, y = cls.split_dataset(dataset, target_index)
        return X, y, classes

    @classmethod
    def split_dataset(cls, dataset, target_index):
        '''
        method to split dataset to X,y and classes
        :param path: url path of dataset
        :param target_index: the index of the target in the ds - default:0
        :return: feature data(X), target data(y), labels of the target(classes)
        '''
        target_index = target_index if target_index > -1 else dataset.shape[1] + target_index

        X = dataset[:, [col_index for col_index in range(dataset.shape[1]) if col_index != target_index]].astype(
            np.float)
        y = dataset[:, [target_index]].astype(np.int)
        return X, y


if __name__ == '__main__':
    path = r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\Ozone Level Detection Data Set\ozone.csv'

    X, y, classes = Parse.read_ds(path)

    print(X)
    print('---------------------------------------------------------------------------')
    print(y)
    print('---------------------------------------------------------------------------')
    print(classes)
