#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 00:13:09 2016

num of dimensions is 32*32*3 so it looks like a 32*32 image with RGB values

@author: yokian
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

oneline = X_test[8]

oneline = sorted(oneline)
portions1 = [oneline[1024*i:1024*(i+1)] for i in range(3)]
portions2 = [oneline[i::3] for i in range(3)]

portion = portions1[0]
modp = portion
#modp /=max(modp)
newp = np.reshape(modp,(32,32))
imgplot = plt.imshow(newp)

#portion = portions2[1]
#modp = portion - min(portion)
##modp /=max(modp)
#newp = np.reshape(modp,(32,32))
#imgplot = plt.imshow(newp)