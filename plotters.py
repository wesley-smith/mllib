"""Some helper functions for machine learning research

Code for many of these routines was taken from *Python Machine Learning - Second Edition*, by Raschka and Mirjalili
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve

from helpers import timing_curve


def plot_means_w_stds(means, stds, xrange, series_labels=None, ylabel=None, xlabel=None, legend=True, linestyles=None, title=None, ylim=None, logx=False, legend_kwargs={}, fig_kwargs={}, markersize=None):
    """Generic plot routine to plot multiple lines on same axes"""

    fig, ax = plt.subplots(**fig_kwargs)

    if not linestyles:
        if len(means) % 2 == 1:
            # ODD sequence
            linestyles = ['-'] * int(len(means))
        else:
            linestyles = ['-'] * int(len(means) / 2) + ['--'] * int(len(means) / 2)
    if not markersize:
        markersize = 5
    if not series_labels:
        series_labels = [''] * len(means)

    for ix, mean, std, label, ls in zip(range(len(means)), means, stds, series_labels, linestyles):
        color = 'C%s' % ix
        plt.plot(xrange, mean, marker='o', markersize=markersize, label=label, color=color, linestyle=ls)
        plt.fill_between(xrange, mean + std, mean - std, color=color, alpha=0.15)

    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    if logx:
        plt.semilogx()
    if legend:
        plt.legend(**legend_kwargs)
    if title:
        plt.title(title)
    plt.show()


def gen_and_plot_validation_curve(estimator, X_train, y_train, param_name, param_range, ylim=None, title=None, **sk_kwargs):
    train_scores, test_scores = validation_curve(
                    estimator=estimator,
                    X=X_train,
                    y=y_train,
                    param_name=param_name,
                    param_range=param_range,
                    **sk_kwargs
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plot_validation_curve(train_mean, train_std, test_mean, test_std, param_name, param_range, ylim, title)

    return train_scores, test_scores, train_mean, train_std, test_mean, test_std


def plot_validation_curve(train_mean, train_std, test_mean, test_std, param_name, param_range, ylim=None, title=None):
    if isinstance(param_range[0], (int,float)):
        x_range = param_range
        x_labels = param_range
    else:
        x_range = range(1, len(param_range) + 1)
        x_labels = [str(p) for p in param_range]

    plt.plot(x_range, train_mean,
             color='C0', marker='o',
             markersize=5, label='training accuracy')
    plt.fill_between(x_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='C0')
    plt.plot(x_range, test_mean,
             color='C1', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(x_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='C1')
    plt.xticks(x_range, x_labels)
    plt.grid()
    plt.legend(loc='lower right')
    plt.xlabel('Parameter %s' % param_name)
    plt.ylabel('Accuracy')
    plt.ylim(ylim)
    plt.title(title)
    plt.show()


def gen_and_plot_learning_curve(estimator, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), ylim=None, ylabel=None, title=None, **kwargs):
    """Plot a learning curve for given estimator"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X_train,
        y=y_train,
        train_sizes=train_sizes,
        **kwargs
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std, ylim, ylabel, title)

    return train_sizes, train_mean, train_std, test_mean, test_std


def plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std, ylim=None, ylabel=None, title=None):


    plt.plot(train_sizes, train_mean,
             color='C0', marker='o',
             markersize=5, label='training')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='C0')

    plt.plot(train_sizes, test_mean,
             color='C1', linestyle='--',
             marker='s', markersize=5,
             label='validation')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='C1')
    plt.grid()
    plt.xlabel('Number of training samples')
    if not ylabel:
        plt.ylabel('Accuracy')
    else:
        plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.ylim(ylim)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion matrix', **kwargs):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Copied from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, **kwargs)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def gen_and_plot_timing_curve(clf, X, y, ylim=None, ylabel=None, title=''):
    df = timing_curve(clf, X, y)
