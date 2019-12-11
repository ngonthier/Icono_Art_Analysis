#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:34:35 2019

@author: gonthier

Inspired from
https://stackoverflow.com/questions/52133285/python-equivalent-fo-hive-numerichistogram/52133807
 Just a mindless copy for easy verification. Not good style or performant.

"""

import random


class Coord:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self): # debug
        return "Coord(" + str(self.x) + ", " + str(self.y) + ")"

    def __str__(self): # debug
        return "Coord(" + str(self.x) + ", " + str(self.y) + ")"

class Random:
    """ This class needs fixin. You'll have to do some work here to make it match your version of Java. """
    def __init__(self, seed):
        random.seed(seed)

    def nextDouble(self):
        return random.uniform(0, 1)

class NumericHistogram:
    def __init__(self):
        self.nbins = 0
        self.nusedbins = 0
        self.bins = None

        self.prng = Random(31183) # This should behave the same as Java's RNG for your Java version.                                                                                                        

    def allocate(self, num_bins):
        self.nbins = num_bins
        self.bins = []
        self.nusedbins = 0

    def add(self, v):
        bin = 0

        l = 0
        r = self.nusedbins

        while(l < r):
            bin = (l+r)//2
            if self.bins[bin].x > v:
                r = bin
            else:
                if self.bins[bin].x < v:
                    l = bin + 1; bin += 1
                else:
                    break

        if bin < self.nusedbins and self.bins[bin].x == v:
            self.bins[bin].y += 1

        else:
            newBin = Coord(x=v, y=1)
            self.bins.append(newBin)
#            if bin == len(self.bins):
#                self.bins.append(newBin)
#                print('append new bins')
#            else:
#                self.bins[bin] = newBin
#                
#                print('first bin')

            self.nusedbins += 1
            if (self.nusedbins > self.nbins):
                self.trim()

                
    def trim(self):
        while self.nusedbins > self.nbins:
            smallestdiff = self.bins[1].x - self.bins[0].x
            smallestdiffloc = 0
            smallestdiffcount = 1
            for i in range(1, self.nusedbins-1):
                diff = self.bins[i+1].x - self.bins[i].x
                if diff < smallestdiff:
                    smallestdiff = diff
                    smallestdiffloc = i
                    smallestdiffcount = 1
                else:
                    smallestdiffcount += 1
                    if diff == smallestdiff and self.prng.nextDouble() <= (1.0/smallestdiffcount):
                        smallestdiffloc = i

            d = self.bins[smallestdiffloc].y + self.bins[smallestdiffloc+1].y
            smallestdiffbin = self.bins[smallestdiffloc]
            smallestdiffbin.x *= smallestdiffbin.y / d
            smallestdiffbin.x += self.bins[smallestdiffloc+1].x / d * self.bins[smallestdiffloc+1].y
            smallestdiffbin.y = d
            self.bins.pop(smallestdiffloc+1)
            self.nusedbins -= 1