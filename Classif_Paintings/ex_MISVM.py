#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 12:40:18 2018

@author:  jnothman gonthier
"""

from __future__ import print_function, division

import numpy as np
import timeit
from milsvm.example.misvmio import parse_c45, bag_set
import misvm
from milsvm import mi_linearsvm

def main():
    # Load list of C4.5 Examples
    example_set = parse_c45('musk1')

    # Group examples into bags
    bagset = bag_set(example_set)

    # Convert bags to NumPy arrays
    # (The ...[:, 2:-1] removes first two columns and last column,
    #  which are the bag/instance ids and class label)
    bags = [np.array(b.to_float())[:, 2:-1] for b in bagset]
    labels = np.array([b.label for b in bagset], dtype=float)
    # Convert 0/1 labels to -1/1 labels
    labels = 2 * labels - 1

    # Spilt dataset arbitrarily to train/test sets
    train_bags = bags[10:]
    train_labels = labels[10:]
    test_bags = bags[:10]
    test_labels = labels[:10]

    # Construct classifiers
    classifiers = {}
#    classifiers['MissSVM'] = misvm.MissSVM(kernel='linear', C=1.0, max_iters=10)
#    classifiers['sbMIL'] = misvm.sbMIL(kernel='linear', eta=0.1, C=1.0)
#    classifiers['SIL'] = misvm.SIL(kernel='linear', C=1.0)
    #classifiers['MISVM'] = misvm.MISVM(kernel='linear', C=1.0, max_iters=10,restarts=0)
    classifiers['MILinearSVM']  = mi_linearsvm.MISVM(C=1.0, max_iters=10,verbose=True,restarts=0)
        
    # Train/Evaluate classifiers
    accuracies = {}
    for algorithm, classifier in classifiers.items():
        print(algorithm)
        start = timeit.default_timer()
        classifier.fit(train_bags, train_labels)
        stop = timeit.default_timer()
        print("Duration :",stop-start)
        predictions = classifier.predict(test_bags)
        accuracies[algorithm] = np.average(test_labels == np.sign(predictions))

    for algorithm, accuracy in accuracies.items():
        print('\n%s Accuracy: %.1f%%' % (algorithm, 100 * accuracy))


if __name__ == '__main__':
    main()