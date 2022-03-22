# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:45:17 2022

@author: jstra
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
        xrel = x-coords[ii][0]+1e-6
        yrel = y-coords[ii][1]+1e-6
        zrel = z-coords[ii][2]+1e-6
        r = (xrel**2 + yrel**2 + zrel**2) ** (3/2)
        es.append(charges[ii]*Vector([xrel/r, yrel/r, zrel/r], units='V/m'))
    return np.sum(es)

def pointchargesV(x,y,z, coords, charges):
    vs = []
    for ii in range(len(charges)):
        xrel = x-coords[ii][0]+1e-6
        yrel = y-coords[ii][1]+1e-6
        zrel = z-coords[ii][2]+1e-6
        r = (xrel**2 + yrel**2 + zrel**2) ** (1/2)
        vs.append(charges[ii]/r)
    return np.sum(vs)





coords = np.array([[-1, 0, 0],
                   [1, -0.5, 0],
                   [1, 0.5, 0],
                   [0, -1, 0]])
charges = [1, 1, -1, -1]

pcv = lambda x,y,z : pointchargesV(x,y,z, coords, charges)
pcdv = lambda x,y,z : -pointcharges(x,y,z, coords, charges)
vv = ScalarField(definition=pcv, graddef=pcdv)

nres = 101
gridres = np.linspace(-3,3,nres)
yy,xx = np.meshgrid(gridres, gridres)

potential = np.zeros((nres,nres))
for ii in range(nres):
    for jj in range(nres):
        potential[ii,jj] = vv((gridres[ii], gridres[jj], 0))

# plt.figure()
vmax = 15
contours = plt.contourf(xx,yy, potential, cmap=cm.seismic, levels=np.linspace(-vmax, vmax, 100), extend='both')
plt.gca().set_aspect('equal')








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

plt.scatter(coords[:,0], coords[:, 1], c=-np.sign(charges), s=np.abs(charges)*100, zorder=100, cmap=cm.coolwarm)

nres = 101
gridres = np.linspace(-3,3,nres)
yy, xx = np.meshgrid(gridres, gridres)

emags = np.zeros((nres,nres))
for ii in range(nres):
    for jj in range(nres):
        emags[ii,jj] = ee((gridres[ii], gridres[jj], 0)).magnitude

# plt.figure()
# plt.contourf(yy,xx,np.log(emags), cmap=cm.plasma, levels=100)
vmax=10
contours = plt.contourf(xx,yy, potential, cmap=cm.plasma, levels=np.linspace(-vmax,vmax, 100), extend='both')







def connected(ix, iy, direction, allnodes):
    try:
        contour = []
        if direction < 0 and ((ix==0) or (iy==0)): # handle bottom and left edges
            return contour
        
        if iy%2==1:     # case of vertical edge
            nxs = [-1,-1,-1] if direction<0 else [0,1,0]
            nys = [-1, 0, 1]
            ndirs = [-1, direction, 1]
        else:           # horizontal edges
            nxs = [0, 0, 1]
            nys = [1, 2, 1] if direction>0 else [-1, -2, -1]
            ndirs = [-1, direction, 1]
            
        for n, (nx, ny, ndir) in enumerate(zip(nxs, nys, ndirs)):
            if allnodes[ix+nx, iy+ny] is not None:
                if n==0:                                              # if first point is a hit, could be saddle point (nodes on all faces)
                    if allnodes[ix+nxs[1], iy+nys[1]] is not None:    # is saddle point
                        print('SADDLE')
                        pass                                          # handle saddle ambiguity
                    else:                                             # otherwise can't be saddle, must be the *only* hit
                        contour.append(allnodes[ix+nx, iy+ny])        # actually append it
                        allnodes[ix+nx, iy+ny] = None                 # clear it so it doesn't get double-counted
                        return contour + connected(ix+nx, iy+ny, ndir, allnodes)   # continue recursion
                else:
                    contour.append(allnodes[ix+nx, iy+ny])        # actually append it
                    allnodes[ix+nx, iy+ny] = None                 # clear it so it doesn't get double-counted
                    return contour + connected(ix+nx, iy+ny, ndir, allnodes)   # continue recursion
        return []
            
            
    
    except IndexError:   # handle right and top edges
        return contour
    
    
    
    
# 2D Marching Squares
        
path = []
res = 28
value= -.2

grid = np.linspace(-3, 3, res)  # base dimensions for grid, hardcoded for now, REVISIT

connections = []    # connectivity between nodes of contour

boxvertices = np.zeros((res, res))        # store potentials at these locations
boxedges = np.full((res+1,res+1), False)  # store whether an edge intersects the contour

allnodes = np.full((res+1, 2*res), None)  # x and y, origin is bottom left, horizontal edges are even y, verts are odd y


# compute values on grid
for ii in range(res):
    for jj in range(res):
        boxvertices[ii,jj] = vv((grid[ii],   grid[jj],   0))
boxpos = np.where((boxvertices-value)>0, 1,0)

# find nodes on edges
for ii in range(res-1):
    for jj in range(res):
        
        # check horizontals
        if boxpos[ii,jj] != boxpos[ii+1,jj]:
            weight = np.abs((boxvertices[ii,jj] - value) / (boxvertices[ii,jj] - boxvertices[ii+1,jj]))
            allnodes[ii, 2*jj] = [grid[ii] * (1-weight) + grid[ii+1]*weight, grid[jj]]
            
        # check verticals
        if boxpos[jj,ii] != boxpos[jj,ii+1]:
            weight = np.abs((boxvertices[jj,ii] - value) / (boxvertices[jj,ii] - boxvertices[jj, ii+1]))
            allnodes[jj, 2*ii+1] = [grid[jj], grid[ii] * (1-weight) + grid[ii+1]*weight]


# # # establish connectivity
# start from the left side, scan up vertically


dots = np.vstack(allnodes[allnodes.nonzero()])


ixs, iys = np.nonzero(allnodes)
contourpaths = []
zzz = allnodes[allnodes.nonzero()[0]]
for ii, (ix,iy) in enumerate(zip(ixs, iys)):
    if np.count_nonzero(allnodes)==0:
        break
    if allnodes[ix, iy] is not None:
        
        point = allnodes[ix, iy]
        
        path1 = None
        path2 = None
        
        try:
            path1 = np.vstack(connected(ix, iy, -1, allnodes))
        except ValueError:
            pass
        
        try:
            path2 = (np.vstack(connected(ix, iy, 1, allnodes)))
        except ValueError:
            pass

        if path1 is None:

            contourpaths.append( np.vstack((point, path2)))
            
        else:
            if path2 is None:
                
                contourpaths.append( np.vstack((path1[::-1,:], point)))
            else:
                contourpaths.append( np.vstack((path1[::-1,:],point, path2)))

        allnodes[ix, iy] = None
                                      
        


xs = dots[:,0]
ys = dots[:,1]
plt.plot(xs,ys, 'r.')


vmax = 5
plt.contourf(xx,yy, potential, cmap=cm.seismic, levels=np.linspace(-vmax, vmax, 100), extend='both')
# plt.contour(xx,yy, potential, colors='k', levels=[value])

for cp in contourpaths:
    plt.plot(cp[:,0], cp[:,1], '-o')

plt.gca().set_aspect('equal')