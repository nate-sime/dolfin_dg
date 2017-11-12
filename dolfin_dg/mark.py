from dolfin import *
import numpy as np


class Marker:

    def mark(self, ind):
        pass


class FixedFractionMarker(Marker):

    def __init__(self, frac=0.1):
        self.frac = frac

    def mark(self, ind):
        assert(isinstance(ind, dolfin.cpp.mesh.CellFunctionDouble))
        assert(ind.dim() == ind.mesh().topology().dim())
        assert(ind.cpp_value_type() == "double")

        # Sort the numpy array of cell function indicators
        idx = np.argsort(-ind.array())

        # Choose only the largest fraction requested TODO: fix for parallel
        idx = idx[0:int(max(self.frac*len(idx), 1))]

        # Populate cell markers
        markers = CellFunction("bool", ind.mesh(), False)
        for i in idx:
            markers[int(i)] = True

        return markers
