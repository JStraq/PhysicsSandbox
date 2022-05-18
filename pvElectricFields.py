# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:45:17 2022

@author: jstra
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d
import itertools as it
from pvcore.pvtypes import *


def pce3d(x, y, z):
    r = (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    return Vector([x / r, y / r, z / r], units='V/m')


def dipole(x, y, z):
    r1 = ((x + 0.5) ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    r2 = ((x - 0.5) ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    e1 = -Vector([(x + 0.5) / r1, y / r1, z / r1], units='V/m')
    e2 = Vector([(x - 0.5) / r2, y / r2, z / r2], units='V/m')
    return e1 + e2


def point_charges(x, y, z, coordinates, charge):
    es = []
    for i in range(len(charge)):
        xrel = x - coordinates[i][0] + 1e-6
        yrel = y - coordinates[i][1] + 1e-6
        zrel = z - coordinates[i][2] + 1e-6
        r = (xrel ** 2 + yrel ** 2 + zrel ** 2) ** (3 / 2)
        es.append(charge[i] * Vector([xrel / r, yrel / r, zrel / r], units='V/m'))
    return np.sum(es)


def point_charges_V(x, y, z, coordinates, charge):
    vs = []
    for i in range(len(charge)):
        xrel = x - coordinates[i][0] + 1e-6
        yrel = y - coordinates[i][1] + 1e-6
        zrel = z - coordinates[i][2] + 1e-6
        r = (xrel ** 2 + yrel ** 2 + zrel ** 2) ** (1 / 2)
        vs.append(charge[i] / r)
    return np.sum(vs)


def connected(xindex, yindex, direction, nodes, scalarfield):
    # try:
    contour = []
    if direction < 0 and ((xindex == 0) or (yindex == 0)):  # handle bottom and left edges
        return contour

    if yindex % 2 == 1:  # case of vertical edge
        nxs = [-1, -1, -1] if direction < 0 else [0, 1, 0]
        nys = [-1, 0, 1]
        ndirs = [-1, direction, 1]
    else:  # horizontal edges
        nxs = [0, 0, 1]
        nys = [1, 2, 1] if direction > 0 else [-1, -2, -1]
        ndirs = [-1, direction, 1]

    for n, (nx, ny, ndir) in enumerate(zip(nxs, nys, ndirs)):
        try:
            if nodes[xindex + nx, yindex + ny] is not None:
                if n == 0:  # if first point is a hit, could be saddle point (nodes on all faces)
                    if nodes[xindex + nxs[1], yindex + nys[1]] is not None:  # is saddle point

                        sidepoint0 = np.array([nodes[xindex + nx, yindex + ny][0],
                                               nodes[xindex + nx, yindex + ny][1],
                                               0])
                        oppopoint = np.array([nodes[xindex + nxs[1], yindex + nys[1]][0],
                                              nodes[xindex + nxs[1], yindex + nys[1]][1],
                                              0])
                        sidepoint1 = np.array([nodes[xindex + nxs[2], yindex + nys[2]][0],
                                               nodes[xindex + nxs[2], yindex + nys[2]][1],
                                               0])
                        centerpoint = (sidepoint0 + sidepoint1) / 2

                        if yindex % 2 == 1:  # vertical
                            cornerpoint = [oppopoint[0], sidepoint0[1], 0]
                        else:  # horizontal
                            cornerpoint = [sidepoint0[0], oppopoint[1], 0]

                        centerval = scalarfield(list(centerpoint))
                        baseval = scalarfield(list(oppopoint))
                        cornerval = scalarfield(list(cornerpoint))

                        if np.sign(centerval - baseval) == np.sign(
                                cornerval - baseval):  # first neighbor point is NOT the right one, switch nx/ny values
                            nx = nxs[2]
                            ny = nys[2]
                            ndir = ndirs[2]

                        contour.append(nodes[xindex + nx, yindex + ny])  # actually append it
                        nodes[xindex + nx, yindex + ny] = None  # clear it so it doesn't get double-counted
                        return contour + connected(xindex + nx, yindex + ny, ndir, nodes, scalarfield)
                        # continue recursion

                    else:  # otherwise, can't be saddle, must be the *only* hit
                        contour.append(nodes[xindex + nx, yindex + ny])  # actually append it
                        nodes[xindex + nx, yindex + ny] = None  # clear it so it doesn't get double-counted
                        return contour + connected(xindex + nx, yindex + ny, ndir, nodes, scalarfield)
                        # continue recursion
                else:
                    contour.append(nodes[xindex + nx, yindex + ny])  # actually append it
                    nodes[xindex + nx, yindex + ny] = None  # clear it so it doesn't get double-counted
                    return contour + connected(xindex + nx, yindex + ny, ndir, nodes, scalarfield)
                    # continue recursion
        except IndexError:
            pass

    return []


# Create some point charges at these locations
coords = np.array([[-1, 0, 0],
                   [1, -0.5, 0],
                   [1, 0.5, 0],
                   [0, -1, 0]])
charges = [1, 1, -1, -1]

# create the associated scalar potential field
vv = ScalarField(definition=lambda x, y, z: point_charges_V(x, y, z, coords, charges),
                 graddef=lambda x, y, z: -point_charges(x, y, z, coords, charges))

# set up the grid to work on
nres = 41
gridres = np.linspace(-3, 3, nres)
yy, xx = np.meshgrid(gridres, gridres)

# calculate the potential at all points
potential = np.zeros((nres, nres))
for ii in range(nres):
    for jj in range(nres):
        potential[ii, jj] = vv((gridres[ii], gridres[jj], 0))

# define the vector field
ee = VectorField(lambda x, y, z: point_charges(x, y, z, coords, charges))

# calculate the electric field magnitudes at all grid points
emags = np.zeros((nres, nres))
for ii in range(nres):
    for jj in range(nres):
        emags[ii, jj] = ee((gridres[ii], gridres[jj], 0)).magnitude

# Implement 2D Marching Squares variant to locate contours

for vi, value in enumerate(np.linspace(np.min(potential)/2, np.max(potential)/2, 30)):
    print(vi)
    plt.figure()
    # value = np.mean(potential)  # not good, misses coverage

    boxedges = np.full((nres + 1, nres + 1), False)  # store whether an edge intersects the contour
    allnodes = np.full((nres + 1, 2 * nres), None)  # x and y, origin is bottom left
    # horizontal edges are even y, verticals are odd y

    boxpos = np.where((potential - value) > 0, 1, 0)  # apply thresholding

    # find nodes on edges
    for ii in range(nres - 1):
        for jj in range(nres):

            # check horizontals
            if boxpos[ii, jj] != boxpos[ii + 1, jj]:
                weight = np.abs((potential[ii, jj] - value) / (potential[ii, jj] - potential[ii + 1, jj]))
                allnodes[ii, 2 * jj] = [gridres[ii] * (1 - weight) + gridres[ii + 1] * weight, gridres[jj]]

            # check verticals
            if boxpos[jj, ii] != boxpos[jj, ii + 1]:
                weight = np.abs((potential[jj, ii] - value) / (potential[jj, ii] - potential[jj, ii + 1]))
                allnodes[jj, 2 * ii + 1] = [gridres[jj], gridres[ii] * (1 - weight) + gridres[ii + 1] * weight]

    ixs, iys = np.nonzero(allnodes)  # indices of the contour nodes

    contourpaths = []
    for ii, (ix, iy) in enumerate(zip(ixs, iys)):
        if np.count_nonzero(allnodes) == 0:
            break
        if allnodes[ix, iy] is not None:

            point = allnodes[ix, iy]
            path1 = None
            path2 = None

            try:
                path1 = np.vstack(connected(ix, iy, -1, allnodes, vv))
            except ValueError:
                pass

            try:
                path2 = (np.vstack(connected(ix, iy, 1, allnodes, vv)))
            except ValueError:
                pass

            if path1 is None:
                contourpaths.append(np.vstack((point, path2)))

            else:
                if path2 is None:
                    contourpaths.append(np.vstack((path1[::-1, :], point)))
                else:
                    contourpaths.append(np.vstack((path1[::-1, :], point, path2)))

            allnodes[ix, iy] = None    # once connected, remove node from consideration

    fluxes = []
    starts = []  # field line starting points, per contour segment

    remnant = []
    for cpath in contourpaths:   # allowing for multiply connected contours
        pointnum = 0
        npoints = len(cpath)
        N = 40

        contourdists = np.linalg.norm(np.diff(cpath, axis=0), axis=1)  # get distances between points
        contourEs = np.array([ee(np.append(point, 0)).magnitude for point in cpath])   # get field magnitude at points
        flux = ((contourEs[:-1] + contourEs[1:]) / 2) * contourdists                   # average field times length

        # filter, integrate, and normalize
        filt = np.exp(-np.linspace(-3, 3, 3)**2)
        flux = np.convolve(flux, filt, mode='same')

        fluxintgrl = np.array([0] + list(it.accumulate(flux)))  # prepend 0
        fluxintgrl *= N/np.max(fluxintgrl)

        # build interpolator object: flux vs step number
        fluxinterp = interp1d(fluxintgrl, np.arange(len(fluxintgrl)))
        positions = fluxinterp(np.arange(N))

        newi = np.floor(positions).astype(int)
        weight = positions % 1
        weightcomp = 1-weight

        newx = (cpath[newi, 0] * weightcomp + cpath[newi + 1, 0] * weight)
        newy = (cpath[newi, 1] * weightcomp + cpath[newi + 1, 1] * weight)
        cstarts = np.vstack((newx, newy))

        starts.append(cstarts.T)

    for cstarts in starts:
        for ii, start in enumerate(cstarts):
            path = ee.streamline(np.append(start, 0))
            if ii == 0:
                plt.plot(path[:, 0], path[:, 1], color='#000000', linestyle='-', linewidth=1)
            else:
                plt.plot(path[:, 0], path[:, 1], color='#AAAAAA', linestyle='-', linewidth=1)

    for ii, cp in enumerate(contourpaths):
        plt.plot(cp[:, 0], cp[:, 1], '.-')

    vmax = 10
    # contours = plt.contourf(xx, yy, potential, cmap=cm.plasma, levels=np.linspace(-vmax, vmax, nres), extend='both')

    plt.gca().set_aspect('equal')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.scatter(coords[:, 0], coords[:, 1], c=-np.sign(charges), s=np.abs(charges) * 100, zorder=100, cmap=cm.coolwarm)
    plt.scatter(coords[:, 0], coords[:, 1], c=-np.sign(charges), s=np.abs(charges) * 100, zorder=100, cmap=cm.coolwarm)

    # plt.show()
    plt.tight_layout()
    plt.savefig('C:/Users/jstra/Desktop/test_{:03d}.png'.format(vi), dpi=300)
    plt.close()
