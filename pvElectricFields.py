# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:45:17 2022

@author: jstra
"""

import numpy as np
import matplotlib.pyplot as plt
from pvcore.pvtypes import *

def pcE3d(x,y,z):
    r = (x**2 + y**2 + z**2) ** (3/2)
    return Vector([x/r, y/r, z/r], units='V/m')

def dipole(x,y,z):
    r1 = ((x+0.5)**2 + y**2 + z**2) ** (3/2)
    r2 = ((x-0.5)**2 + y**2 + z**2) ** (3/2)
    e1 = -Vector([(x+0.5)/r1, y/r1, z/r1], units='V/m')
    e2 = Vector([(x-0.5)/r2, y/r2, z/r2], units='V/m')
    return e1+e2

def pointcharges(x,y,z, coords, charges):
    es = []
    for ii in range(len(charges)):
        xrel = x-coords[ii][0]
        yrel = y-coords[ii][1]
        zrel = z-coords[ii][2]
        r = (xrel**2 + yrel**2 + zrel**2) ** (3/2)
        es.append(charges[ii]*Vector([xrel/r, yrel/r, zrel/r], units='V/m'))
    return np.sum(es)


# ee = VectorField(pcE3d)
# ee = VectorField(dipole)

coords = np.array([[-1, 0, 0],
                   [1, -0.5, 0],
                   [1, 0.5, 0],
                   [0, -1, 0]])
charges = [1,-1, -1, 5]

starts = []


######################################################################################
# START POINTS AROUND EACH CHARGE
# ns = 8
# srad = 0.05
# for ii in range(len(charges)):
#     for n in range(int(np.abs(charges[ii]*ns))):
#         starts.append([coords[ii][0] + srad*np.cos(2*np.pi*n/(np.abs(charges[ii]*ns))),
#                        coords[ii][1] + srad*np.sin(2*np.pi*n/(np.abs(charges[ii]*ns))),
#                        coords[ii][2]])

# START POINTS BASED ON EXTERNAL RADIUS: GAUSS'S LAW KIND OF ARGUMENT
srad = 2
ns = 23
for n in range(ns):
    starts.append([srad*np.cos(2*np.pi*n/ns),
                   srad*np.sin(2*np.pi*n/ns),
                   0])
    
# NOT HAPPY WITH EITHER OF THESE:
# start points at charge works well for dipole case, not well for multipoles
# starts around outside work well when all charges have same sign, loses internal structure for multipoles


# is there a way to scale line density to true field magnitude?
#######################################################################################
    
pc = lambda x,y,z : pointcharges(x,y,z, coords, charges)
ee = VectorField(pc)

# start=[0.01, 0.1, 0]
# ee(start)

for start in starts:
    path = ee.streamline(start)
    plt.plot(path[:,0], path[:,1], color='#AAAAAA', linestyle='-', linewidth=1)
    
starts = np.array(starts)
# plt.scatter(starts[:,0], starts[:,1], c='r')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.gca().set_aspect('equal')

plt.scatter(coords[:,0], coords[:, 1], c='r', s=np.abs(charges)*100, zorder=100)