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


class FixedFractionMarkerParallel(Marker):

    def __init__(self, frac=0.1):
        self.frac = frac

    def mark(self, ind):
        assert(isinstance(ind, dolfin.cpp.mesh.CellFunctionDouble))
        assert(ind.dim() == ind.mesh().topology().dim())
        assert(ind.cpp_value_type() == "double")

        comm = ind.mesh().mpi_comm().tompi4py()
        ind_array = comm.gather(ind.array(), 0)

        if comm.rank == 0:
            offsets = np.cumsum(list(map(len, ind_array)))
            assert(len(offsets) == comm.size)
            ind_array = np.hstack(ind_array)

            # Sort the numpy array of cell function indicators
            idx = np.argsort(-ind_array)

            # Choose only the largest fraction requested
            idx = idx[0:int(max(self.frac*len(idx), 1))]

            def owning_process(idx):
                for p in range(comm.size):
                    if idx < offsets[p]:
                        return p

            comm_back = [[] for p in range(comm.size)]
            for i in idx:
                p = owning_process(i)
                comm_back[p].append(i - offsets[p-1] if p > 0 else i)
        else:
            comm_back = None

        idx = comm.scatter(comm_back)

        # Populate cell markers
        markers = CellFunction("bool", ind.mesh(), False)
        for i in idx:
            markers[int(i)] = True

        return markers
