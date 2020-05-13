# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

data_file = np.loadtxt('C:/Users/giorgos/Desktop/inliers.txt', delimiter=',')
data_file2 = np.loadtxt('C:/Users/giorgos/Desktop/out.txt', delimiter=',')

data_file_c0 = np.loadtxt('C:/Users/giorgos/Desktop/c0.txt', delimiter=',')
data_file_c1 = np.loadtxt('C:/Users/giorgos/Desktop/c1.txt', delimiter=',')
data_file_c2 = np.loadtxt('C:/Users/giorgos/Desktop/c2.txt', delimiter=',')
data_file_c3 = np.loadtxt('C:/Users/giorgos/Desktop/c3.txt', delimiter=',')
data_file_c4 = np.loadtxt('C:/Users/giorgos/Desktop/c4.txt', delimiter=',')

axisX = data_file[:,0]
axisY = data_file[:,1]
outX = data_file2[:,0]
outY = data_file2[:,1]

c0X = data_file_c0[:,0]
c0Y = data_file_c0[:,1]
c1X = data_file_c1[:,0]
c1Y = data_file_c1[:,1]
c2X = data_file_c2[:,0]
c2Y = data_file_c2[:,1]
c3X = data_file_c3[:,0]
c3Y = data_file_c3[:,1]
c4X = data_file_c4[:,0]
c4Y = data_file_c4[:,1]
"""
cluster = data_file[:,2]
outlier = data_file[:,3]

plt.plot(axisX,axisY)
plt.axis(0, 5000, 0, 5000)
plt.show()
"""

plt.plot(axisX,axisY, 'go')
plt.axis([0, 6, 0, 1100])
plt.plot(outX,outY, 'ro')
plt.axis([0, 6, 0, 1100])
plt.savefig("in and out.jpg", dpi=1500)
plt.show()
"""
plt.plot(c0X,c0Y, 'go')
plt.axis([0, 6, 0, 1100])
plt.plot(c1X,c1Y, 'ro')
plt.axis([0, 6, 0, 1100])
plt.plot(c2X,c2Y, 'bo')
plt.axis([0, 6, 0, 1100])
plt.plot(c3X,c3Y, 'mo')
plt.axis([0, 6, 0, 1100])
plt.plot(c4X,c4Y, 'yo')
plt.axis([0, 6, 0, 1100])
plt.savefig("clusters.jpg", dpi=1500)
plt.show()
"""