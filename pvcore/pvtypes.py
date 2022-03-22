# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:44:30 2022

@author: jstra
"""
import numpy as np

class Vector:
    def __init__(self, components, units=None):
        self.components = np.array(components)
        self.units = units
        
    def __str__(self):
        if self.units is None:
            return f"vector {tuple(self.components)}"
        else:
            return "vector ({:s}) {:s}".format(", ".join(['{:.3g}'.format(x) for x in self.components]), self.units)
        
    def __repr__(self):
        if self.units is None:
            return f"{self.dimension}-dimensional vector (unitless)"
        else:
            return f"{self.dimension}-dimensional vector ({self.units})"
        
    def __mul__(self, other):
        if isinstance(other, (float, int)):  # just scale the vector, return another vector
            return Vector(self.components*other, units=self.units)
        
        elif isinstance(other, np.ndarray):
            new = np.array([x*self.components for x in other])
            return new
        
        else:
            raise TypeError(f"cannot multiply {self!r} by {other!r} of type {type(other)}")
    
    def __rmul__(self, other):
        if isinstance(other, (float, int)):  # just scale the vector, return another vector
            return Vector(self.components*other, units=self.units)
        
        elif isinstance(other, np.ndarray):
            new = np.array([x*self.components for x in other])
            return new
        
        else:
            raise TypeError(f"cannot multiply object {other!r} of type {type(other)} by Vector object")
            
    def __add__(self, other):
        if isinstance(other, Vector):
            if self.dimension == other.dimension:
                if self.units == other.units:
                    return Vector(self.components+other.components, units=self.units)
                else:
                    raise TypeError(f"cannot add {self!r} to {other!r}")
            else:
                raise TypeError(f"cannot add {self!r} to {other!r}")
        else:
            raise TypeError(f"cannot add {self!r} to object {other} of type {type(other)}")
            
    def __radd__(self, other):
        return self.__add__(self, other)
    
    def __getitem__(self, ii):
        return self.components[ii]
    
    def __neg__(self):
        return Vector(components=-self.components, units=self.units)
    
    @property
    def dimension(self):
        return len(self.components)
    
    @property
    def magnitude(self):
        return np.sqrt( np.dot(self.components, self.components) )
    
    @property
    def direction(self):
        return Vector(self.components / self.magnitude)




class VectorField():
    def __init__(self, definition=None, units=None, coordinates=None, components=None):
        self.deftype = "function" if definition is not None else "data"
        self.definition = definition
        self.units = units
        self.components = components
        self.coordinates = coordinates

    def __call__(self, point, *args):
        if self.deftype == "function":
            return self.definition(*point, *args)
        
    def streamline(self, start, maxcoords=None, mincoords=None):
        
        N = len(start)      
        paths = [[start], [start]]
        
        maxcoords = 5*np.ones(N) if maxcoords is None else maxcoords
        mincoords = -5*np.ones(N) if mincoords is None else mincoords
        maxsteps = 1000
        minthresh = 1e-5
        maxthresh = 1e5
        minstep = 0.01
        maxstep = 0.1
        
        
        for jj, sign in enumerate([-1,1]):
            ctr = 0
            terminate = False
            cosines = []
            thisvec = self(start)
            oldvec = self(start)
        
            while not terminate:
                thisvec = self(paths[jj][-1])
                mag = thisvec.magnitude
                step = sign * min(maxstep, max(minstep, 1/mag))
                paths[jj].append([paths[jj][-1][ii] + step * thisvec.direction[ii] for ii in range(N)])

                if (mag < minthresh) or (mag) > maxthresh:
                    # print(sign, 'mag', mag)
                    terminate = True
                if any([(paths[jj][-1][ii]>maxcoords[ii]) or (paths[jj][-1][ii]<mincoords[ii]) for ii in range(N)]):
                    # print(sign, 'coord', paths[jj][-1])
                    terminate = True
                if np.nan in paths[jj][-1]:
                    # print(sign, 'nan')
                    terminate = True
                
                ctr += 1
                if ctr >= maxsteps:
                    # print(sign, 'steps', ctr)
                    terminate = True
                oldvec = thisvec
                
        return np.vstack((paths[1][::-1], paths[0]))
    
    
    
class ScalarField():
    def __init__(self, definition=None, graddef=None, units=None, coordinates=None, values=None):
        self.deftype = "function" if definition is not None else "data"
        self.gradtype = "function" if graddef is not None else "manual"
        self.definition = definition
        self.graddef = graddef
        self.units = units
        self.values = values
        if self.values is None:
            self.values = {}
            self.coordinates = []
        else:
            self.values = values
            self.coordinates = coordinates
        
        ## NEED ARBITRARY DIMENSIONS SUPPORT?

    def __call__(self, point, *args):
        if self.deftype == "function":
            if point not in self.coordinates:
                self.coordinates.append(point)
                self.values[tuple(point)] = self.definition(*point, *args)
            return self.values[tuple(point)]
        
    def gradient(self, coordinates):
        if self.gradtype == "function":
            return self.graddef(coordinates)
        
    def streamline(self, start, maxcoords=None, mincoords=None):
        raise NotImplementedError()
        
    def contour(self, values, maxcoords=None, mincoords=None, res=None):
        """
        Implement marching squares with a linear interpolation to find specified contours
        """
        
        # hardcode limits for dev
        # start with 2D case
        
        path = []
        
        grid = np.linspace(-3, 3, res)
        for ii in range(res-1):
            for jj in range(res-1):
                
                box = np.zeros((2,2))
                box[0,0] = self((grid[ii],   grid[jj]  ))  # left bottom
                box[0,1] = self((grid[ii],   grid[jj+1]))  # left top
                box[1,0] = self((grid[ii+1], grid[jj]  ))  # right bottom
                box[1,1] = self((grid[ii+1], grid[jj+1]))  # right top
                
                boxpos = np.where((box-value)>0, 1,0)