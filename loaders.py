"""Data loading routines

Sources:
- https://www.valentinmihov.com/2015/04/17/adult-income-data-set/
"""

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