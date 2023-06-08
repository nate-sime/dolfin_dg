import dolfinx
import numpy as np


def filter_over_celltags(ind: dolfinx.mesh.MeshTags,
                         function: "(np.ndarray) -> np.ndarray") -> np.ndarray:
    """
    Filter a given measure defined on all cells in cell tags returning
    qualifying cell indices. This is particularly used in measures of cellwise
    error indicators.

    Parameters
    ----------
    ind
        Cell tags of floats
    function
        Given cell tag values, return those indices which are to be returned
        to the owning process

    Returns
    -------
        Qualifying cell indices

    """
    assert(ind.dim == ind.topology.dim)

    # Communicate all the indicator values to process 0
    # It is important to preserve their order
    comm = ind.topology.comm
    ind_array = comm.gather(ind.values, 0)

    if comm.rank == 0:
        # On process 0 we argsort the indicators to find
        # the indices of the cells with the `frac' largest
        # error estimates.

        # The offsets are the global element index offsets assigned
        # to each process
        offsets = np.cumsum(list(map(len, ind_array)))
        assert(len(offsets) == comm.size)

        idx = function(ind_array)

        # Utility function to find which process owns a global cell index
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
    return ind.indices[idx]


def maximal_indices_fraction(ind: dolfinx.mesh.MeshTags, fraction: float):
    """
    Compute the cell indices corresponding to the largest values in
    the provided cell tags, where the number of cell indices returned is
    the given fraction of the total number of cells in the mesh.

    Parameters
    ----------
    ind
        Cell tags of floats
    fraction
        Fraction of total number of cells

    Returns
    -------
        Fraction of mesh total cells' indices which are largest in the
        provided cell tags

    """

    def function(ind_array):
        # Sort the numpy array of cell function indicators
        ind_array = np.hstack(ind_array)
        idx = np.argsort(-ind_array)

        # Choose only the largest fraction requested
        idx = idx[0:int(max(fraction * len(idx), 1))]
        return idx

    return filter_over_celltags(ind, function)
