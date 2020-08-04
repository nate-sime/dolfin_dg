import dolfin
import numpy as np


class Marker:

    def mark(self, ind):
        pass


class FixedFractionMarker(Marker):

    def __init__(self, frac=0.1):
        self.frac = frac

    def mark(self, ind):
        assert(ind.dim() == ind.mesh().topology().dim())
        assert(isinstance(ind, dolfin.cpp.mesh.MeshFunctionDouble))

        # Sort the numpy array of cell function indicators
        idx = np.argsort(-ind.array())

        # Choose only the largest fraction requested
        idx = idx[0:int(max(self.frac*len(idx), 1))]

        # Populate cell markers
        markers = dolfin.MeshFunction("bool", ind.mesh(),
                                      ind.mesh().topology().dim(),
                                      False)
        for i in idx:
            markers[int(i)] = True

        return markers


class FixedFractionMarkerParallel(Marker):

    def __init__(self, frac=0.1):
        self.frac = frac

    def mark(self, ind):
        assert(ind.dim() == ind.mesh().topology().dim())
        assert(isinstance(ind, dolfin.cpp.mesh.MeshFunctionDouble))

        # Communicate all of the indicator values to process 0
        # It is important to preserve their order
        comm = ind.mesh().mpi_comm()
        ind_array = comm.gather(ind.array(), 0)

        if comm.rank == 0:
            # On process 0 we argument sort the indicators to find
            # the indices of the cells with the `frac' largest
            # error estimates.

            # The offsets are the global element index offsets assigned
            # to each process
            offsets = np.cumsum(list(map(len, ind_array)))
            assert(len(offsets) == comm.size)

            # Sort the numpy array of cell function indicators
            ind_array = np.hstack(ind_array)
            idx = np.argsort(-ind_array)

            # Choose only the largest fraction requested
            idx = idx[0:int(max(self.frac*len(idx), 1))]

            # Utiltiy functino to find which process owns a global cell index
            def owning_process(idx):
                for p in range(comm.size):
                    if idx < offsets[p]:
                        return p

            # Generate a list of the cell indices to be refined and communicate
            # them back to their process owners
            comm_back = [[] for p in range(comm.size)]
            for i in idx:
                p = owning_process(i)
                comm_back[p].append(i - offsets[p-1] if p > 0 else i)
        else:
            comm_back = None

        idx = comm.scatter(comm_back)

        # Populate cell markers
        markers = dolfin.MeshFunction("bool", ind.mesh(),
                                      ind.mesh().topology().dim(), False)
        for i in idx:
            markers[int(i)] = True

        return markers
