import enum
import dolfinx
from petsc4py import PETSc


class MatrixType(enum.Enum):
    monolithic = enum.auto()
    block = enum.auto()
    nest = enum.auto()

    def is_block_type(self):
        return self is not MatrixType.monolithic


class NonlinearPDE_SNESProblem():
    def __init__(self, F, J, soln_vars, bcs, P=None):
        self.L = F
        self.a = J
        self.a_precon = P
        self.bcs = bcs
        self.soln_vars = soln_vars

    def F_mono(self, snes, x, F):
        x.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with x.localForm() as _x, self.soln_vars.vector.localForm() as _u:
            _u[:] = _x
        with F.localForm() as f_local:
            f_local.set(0.0)
        dolfinx.fem.assemble_vector(F, self.L)
        dolfinx.fem.apply_lifting(F, [self.a], [self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(F, self.bcs, x, -1.0)

    def J_mono(self, snes, x, J, P):
        J.zeroEntries()
        dolfinx.fem.assemble_matrix(J, self.a, self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfinx.fem.assemble_matrix(
                P, self.a_precon, self.bcs, diagonal=1.0)
            P.assemble()

    def F_block(self, snes, x, F):
        assert x.getType() != "nest"
        assert F.getType() != "nest"
        x.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)

        offset = 0
        for var in self.soln_vars:
            size_local = var.vector.getLocalSize()
            var.vector.array[:] = x.array_r[offset:offset + size_local]
            var.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset += size_local

        dolfinx.fem.assemble_vector_block(
            F, self.L, self.a, self.bcs, x0=x, scale=-1.0)

    def J_block(self, snes, x, J, P):
        assert x.getType() != "nest" and J.getType() != "nest" \
               and P.getType() != "nest"
        J.zeroEntries()
        dolfinx.fem.assemble_matrix_block(J, self.a, self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfinx.fem.assemble_matrix_block(
                P, self.a_precon, self.bcs, diagonal=1.0)
            P.assemble()

    def F_nest(self, snes, x, F):
        assert x.getType() == "nest" and F.getType() == "nest"
        # Update solution
        x = x.getNestSubVecs()
        for x_sub, var_sub in zip(x, self.soln_vars):
            x_sub.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            with x_sub.localForm() as _x, var_sub.vector.localForm() as _u:
                _u[:] = _x

        # Assemble
        bcs1 = dolfinx.cpp.fem.bcs_cols(
            dolfinx.fem.assemble._create_cpp_form(self.a), self.bcs)
        for L, F_sub, a, bc in zip(self.L, F.getNestSubVecs(), self.a, bcs1):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            dolfinx.fem.assemble_vector(F_sub, L)
            dolfinx.fem.apply_lifting(F_sub, a, bc, x0=x, scale=-1.0)
            F_sub.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = dolfinx.cpp.fem.bcs_rows(
            dolfinx.fem.assemble._create_cpp_form(self.L), self.bcs)
        for F_sub, bc, x_sub in zip(F.getNestSubVecs(), bcs0, x):
            dolfinx.fem.set_bc(F_sub, bc, x_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def J_nest(self, snes, x, J, P):
        assert x.getType() == "nest" and J.getType() == "nest" \
               and P.getType() == "nest"
        J.zeroEntries()
        dolfinx.fem.assemble_matrix_nest(J, self.a, self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfinx.fem.assemble_matrix_nest(
                P, self.a_precon, self.bcs, diagonal=1.0)
            P.assemble()
