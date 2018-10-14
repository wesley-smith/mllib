"""Data loading routines

Sources:
- https://www.valentinmihov.com/2015/04/17/adult-income-data-set/
"""


import os
import glob

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelEncoder


def load_adult(preprocess=False):
    df = pd.read_csv(
        './mllib/data/adult.csv',
        names=[
            "age", "workclass", "fnlwgt", "education", "education-num", "marital status",
            "occupation", "relationship", "race", "sex", "capital gain", "capital loss",
            "hours per week", "country", "target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

    if preprocess:
        # Remove missing data
        df = df.dropna()

        # Perform one hot encoding
        df = pd.get_dummies(df)

        # Want the target to be 0 or 1
        df['target'] = df['target_>50K']
        del df["target_<=50K"]
        del df["target_>50K"]


    print('Dataset shape %s' % str(df.shape))
    print('Value composition:\n%s' % (df['target'].value_counts() / df['target'].shape[0]))

    return df


def load_mnist(preprocess=False):
    """

    :param preprocess: filter data, keeping only labels 4 and 9, and encode labels (0==4, 1==9)
    :type preprocess: bool
    :return:
    :rtype:
    """
    mnist = fetch_mldata('MNIST original', data_home='./mllib/data/')  # Data_home is a cache directory
    raw_df = pd.DataFrame(mnist.data)
    raw_df['target'] = mnist.target
    if not preprocess:
        return raw_df

    criterion = raw_df['target'].map(lambda x: x in [4, 9])
    filtered_df = raw_df[criterion]
    le = LabelEncoder()
    filtered_df['target'] = le.fit_transform(filtered_df['target'])
    print('Target column encoded with the following classes %s' % le.classes_)
    print('Dataset shape %s' % str(filtered_df.shape))
    print('Value composition:\n%s' % (filtered_df['target'].value_counts() / filtered_df['target'].shape[0]))

    return filtered_df


def read_experiment_results(*, directory, prefix, suffix, param_names):
    filenames = [fn for fn in os.listdir(directory) if fn.startswith(prefix) and fn.endswith(suffix)]
    paths = [os.path.join(directory, fn) for fn in filenames]
    paramsets = [fn[len(prefix):-(len(suffix))].split('_') for fn in filenames]
    result_df = pd.DataFrame()
    for path, paramset in zip(paths, paramsets):
        df = pd.read_csv(path)
        for param, param_name in zip(paramset, param_names):
            try:
                df[param_name] = int(param)
            except ValueError:
                df[param_name] = float(param)
        result_df = pd.concat([result_df, df])
    return result_df


"""@author: Farrukh Rahman

"""

def parse_experiment(filename, start_str='CONTPEAKS_', end_str='_LOG.txt', split_idx=1):
    end_len = len(end_str)
    return filename.split(start_str)[split_idx][:-end_len]


def read_experiment(filename, experiment_field='experiment', start_str='CONTPEAKS_', end_str='_LOG.txt', split_idx=1):
    df = pd.read_csv(filename)
    df[experiment_field] = parse_experiment(filename, start_str=start_str, end_str=end_str, split_idx=split_idx)
    return df


def concat_files(directory='CONTPEAKS',
                 filename_string='CONTPEAKS_*.txt', start_str=None, end_str='_LOG.txt', split_idx=1):
    """[loop through files, check whether they match the relevent pattern, consolidate the ones that do into a dataframe with experiment fieldname]

    Keyword Arguments:
        filename_string {str} -- [pattern to match files in directory] (default: {'CONTPEAKS_RHC*.txt'})
        start_str {str} -- [start string. the string between start and stop is designated as experiment field name] (default: {'CONTPEAKS_'})
        end_str {str} -- [stop string] (default: {'_LOG.txt'})
        split_idx {int} -- [denotes which index of subtring to be parsed] (default: {1})

    Returns:
        [dataframe] -- [consolidates files into dataframe and adds experiment field]
    """
    if not start_str:
        start_str = directory
    files = glob.glob(os.path.join(directory, filename_string))
    print(files)
    df = pd.concat([read_experiment(fp, start_str=start_str, end_str=end_str, split_idx=split_idx) for fp in files],
                   ignore_index=True)
    return df