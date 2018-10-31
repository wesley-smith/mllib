from collections import defaultdict, Counter
from datetime import datetime
import os
from pathlib import Path
import pickle
from time import clock

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.training_utils import multi_gpu_model
import numpy as np
import pandas as pd
from pytz import timezone
from sklearn.base import clone
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight


# Keras NN features
def build_keras_clf(n_input_features=1, hidden_layer_sizes=(10, 10), learning_rate_init=0.01, momentum=0.8, n_gpus=0):
    """This function builds a Keras model for use with scikit's GridSearch"""
    if not isinstance(hidden_layer_sizes, tuple):
        hidden_layer_sizes = (hidden_layer_sizes,)

    model = Sequential()

    model.add(Dense(units=n_input_features, input_shape=(n_input_features,), activation='relu'))

    for layer_size in hidden_layer_sizes:
        assert layer_size > 0
        model.add(Dense(units=layer_size, activation='relu'))

    # Add output layer
    model.add(Dense(units=1, activation='sigmoid'))

    # Set the number of GPUs
    if n_gpus > 1:
        model = multi_gpu_model(model, gpus=n_gpus)

    sgd = SGD(lr=learning_rate_init, momentum=momentum, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])

    return model


def save_cluster_result(results, dataset, algorithm, extras=''):
    filepath = 'results/%s_%s_%s.pkl' % (dataset, algorithm, extras)
    save_to_file(results, filepath)


def load_cluster_result(dataset, algorithm, extras=None):
    if extras:
        filepath = 'results/%s_%s_%s.pkl' % (dataset, algorithm, extras)
    else:
        filepath = 'results/%s_%s.pkl' % (dataset, algorithm)
    return load_from_file(filepath)


def save_to_file(obj, filepath):
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_file(filepath):
    with open(filepath, 'rb') as fp:
        result = pickle.load(fp)
    return result


def save_search_result(search_obj, dataset, estimator_type, results_dir='./results/', extras=''):
    """Saves a GridSearchCV or RandomizedSearchCV result to a file"""
    results = search_obj.cv_results_
    score = search_obj.best_score_
    tz = timezone('US/Eastern')
    date = datetime.now(tz).isoformat(timespec='minutes', sep='_').replace(':', '-')
    filename = 'search_%s_%s_%.3f_%s_%s' % (dataset, estimator_type, score, date, extras)
    path = Path(results_dir + filename + '.pkl')
    joblib.dump(results, path)


def save_learning_curve(dataset, learner_type, train_sizes, train_mean, train_std, test_mean, test_std, results_dir='./results/', extras=''):
    learning_result = {
        'train_sizes': train_sizes,
        'train_mean': train_mean,
        'train_std': train_std,
        'test_mean': test_mean,
        'test_std': test_std,
    }
    score = np.max(learning_result['test_mean'])
    tz = timezone('US/Eastern')
    date = datetime.now(tz).isoformat(timespec='minutes', sep='_').replace(':', '-')
    filename = 'learning_%s_%s_%.3f_%s_%s' % (dataset, learner_type, score, date,extras)
    path = Path(results_dir + filename + '.pkl')
    joblib.dump(learning_result, path)


def load_best_search(dataset, estimator_type, results_dir='./results/'):
    """Loads the best (highest scoring) result for a previously-executed grid/randomized search"""
    results_files = [fn for fn in os.listdir(results_dir)
                     if fn.startswith('search') and
                     estimator_type in fn and dataset in fn]
    best_results_file = sorted(results_files)[-1]
    print('Found the following results files for this dataset/algorithm: %s' % results_files)
    print('Returning results for the highest-scoring-file: %s' % best_results_file)
    return joblib.load(results_dir + best_results_file)


def load_best_learning(dataset, estimator_type, results_dir='./results/'):
    """Loads the best (highest scoring) result for a previously-executed learning curve"""
    results_files = [fn for fn in os.listdir(results_dir)
                     if fn.startswith('learning') and
                     estimator_type in fn and dataset in fn]
    best_results_file = sorted(results_files)[-1]
    print('Found the following results files for this dataset/algorithm: %s' % results_files)
    print('Returning results for the highest-scoring-file: %s' % best_results_file)
    return joblib.load(results_dir + best_results_file)


def load_result_by_name(filename, results_dir='./results/'):
    return joblib.load(results_dir + filename)


def scikit_cv_result_to_df(cv_res, drop_splits=True):
    """Convert a GridSearchCV.cv_result_ dictionary to a dataframe indexed by hyperparameter

    :param cv_res: cv_result_ object (attribute of a GridSearchCV/RandomizedSearchCV instance)
    :type cv_res: dict
    :param drop_splits: whether to drop columns with individual cross-validation scores (default: True)
    :type drop_splits: bool
    :return: DataFrame with a MultiIndex corresponding to each parameter
    :rtype: pd.DataFrame
    """
    params = [k for k in cv_res.keys() if k.startswith('param_')]
    params_to_shorthand = { # Mapping type to remove the 'param_' prefix
        p : p[6:] for p in params
    }
    cv_res_df = pd.DataFrame(cv_res, columns=[k for k in cv_res.keys() if k != 'params'])
    cv_res_df = cv_res_df.rename(index=str, columns=params_to_shorthand)
    cv_res_df = cv_res_df.set_index(list(params_to_shorthand.values()))
    if drop_splits:
        cv_res_df = cv_res_df.drop(axis=1, labels=[c for c in cv_res_df.columns if c.startswith('split')])
    return cv_res_df


# All code below copied from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/helpers.py
def balanced_accuracy(truth,pred):
    wts = compute_sample_weight('balanced',truth)
    return accuracy_score(truth,pred,sample_weight=wts)

balanced_accuracy_scorer = make_scorer(balanced_accuracy)


# This code is modified so that this classifier is cloned before each fit, avoiding timing advantages
def timing_curve(clf, X, Y, train_sizes=np.linspace(0.1, 0.9, 9)):
    out = {
        'train_size': [],
        'fit_time': [],
        'pred_time':  [],
    }
    for frac in train_sizes:
        clf_clone = clone(clf)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=frac, random_state=1)
        out['train_size'].append(X_train.shape[0])
        st = clock()
        np.random.seed(55)
        clf_clone.fit(X_train,y_train)
        out['fit_time'].append(clock()-st)
        st = clock()
        clf_clone.predict(X_test)
        out['pred_time'].append(clock()-st)
    out_df = pd.DataFrame(out)
    out_df['fit_time_per_samp'] = out_df['fit_time'] / out_df['train_size']
    out_df['pred_time_per_samp'] = out_df['pred_time'] / out_df['train_size']
    out_df = out_df.set_index('train_size')
    return out_df


def save_timing_curve(timing_df, dataset, estimator_type, results_dir='./results/', extras=''):
    """Saves a GridSearchCV or RandomizedSearchCV result to a file"""
    tz = timezone('US/Eastern')
    date = datetime.now(tz).isoformat(timespec='minutes', sep='_').replace(':', '-')
    filename = 'timing_%s_%s_%s_%s' % (dataset, estimator_type, date, extras)
    path = Path(results_dir + filename + '.pkl')
    joblib.dump(timing_df, path)


def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
#    assert max(pred) == max(Y)
#    assert min(pred) == min(Y)
    return accuracy_score(Y,pred)