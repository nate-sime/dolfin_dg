import enum
import dolfinx
from petsc4py import PETSc


class MatrixType(enum.Enum):
    monolithic = enum.auto()
    block = enum.auto()
    nest = enum.auto()


class GenericSNESProblem():
    def __init__(self, a, L, P, bcs, soln_vars,
                 assemble_type=MatrixType.monolithic,
                 use_preconditioner=False):
        if not assemble_type is MatrixType.monolithic:
            assert isinstance(a, list)
            assert isinstance(L, list)
        self.L = L
        self.a = a
        self.P = P
        if not hasattr(bcs, "__len__"):
            bcs = [bcs]
        self.bcs = bcs
        # self.soln_vec = soln_vec
        self.soln_vars = soln_vars
        self.assemble_type = assemble_type
        self.use_preconditioner = use_preconditioner

        if assemble_type is MatrixType.monolithic:
            self.F = self.F_mono
            self.J = self.J_mono
        elif assemble_type is MatrixType.block:
            self.F = self.F_block
            self.J = self.J_block
        elif assemble_type is MatrixType.nest:
            self.F = self.F_nest
            self.J = self.J_nest

    def F_mono(self, snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.soln_vars.vector)
        self.soln_vars.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        dolfinx.fem.assemble_vector(F, self.L)
        dolfinx.fem.apply_lifting(F, a=[self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(F, self.bcs, x, -1.0)

    def J_mono(self, snes, x, J, P):
        J.zeroEntries()
        dolfinx.fem.assemble_matrix(J, self.a, bcs=self.bcs)
        J.assemble()

        if self.use_preconditioner:
            P.zeroEntries()
            dolfinx.fem.assemble_matrix(P, self.P, bcs=self.bcs)
            P.assemble()
        # self._J.view()

    def F_block(self, snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        offset = 0
        for var in self.soln_vars:
            size_local = var.vector.getLocalSize()
            var.vector.array[:] = x.array_r[offset:offset + size_local]
            var.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset += size_local

        with F.localForm() as f_local:
            f_local.set(0.0)

        dolfinx.fem.assemble_vector_block(F, self.L, self.a, self.bcs, x0=x, scale=-1.0)

    def J_block(self, snes, x, J, P):
        J.zeroEntries()
        dolfinx.fem.assemble_matrix_block(J, self.a, self.bcs, 1.0)
        J.assemble()

        if self.use_preconditioner:
            P.zeroEntries()
            dolfinx.fem.assemble_matrix_block(P, self.P, self.bcs, 1.0)
            P.assemble()

    def F_nest(self, snes, x, F):
        for x_sub, var_sub in zip(x.getNestSubVecs(), self.soln_vars):
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            with x_sub.localForm() as _x, var_sub.vector.localForm() as _u:
                _u[:] = _x

        dolfinx.fem.assemble_vector_nest(F, self.L)

        dolfinx.fem.assemble.apply_lifting_nest(F, self.a, self.bcs, x0=x, scale=-1.0)
        for F_sub in F.getNestSubVecs():
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bcs0 = dolfinx.cpp.fem.bcs_rows(dolfinx.fem.assemble._create_cpp_form(self.L), self.bcs)
        dolfinx.fem.assemble.set_bc_nest(F, bcs0, x0=x, scale=-1.0)

        F.assemble()

    def J_nest(self, snes, x, J, P):
        J.zeroEntries()
        diagonal = 1.0
        dolfinx.fem.assemble_matrix_nest(J, self.a, self.bcs, diagonal)
        J.assemble()

        if self.use_preconditioner:
            P.zeroEntries()
            fem.assemble_matrix_nest(P, self.P, self.bcs, diagonal)
            P.assemble()