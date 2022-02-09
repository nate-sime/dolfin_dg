from petsc4py import PETSc
import ufl
import dolfinx.fem


def _form_with_estimated_quad_degree(mesh, form, quadrature_degree):
    if quadrature_degree is None:
        quadrature_degree = \
            mesh.ufl_domain().ufl_coordinate_element().degree() + 1

    dolfinx_form = dolfinx.fem.form(
        form, form_compiler_parameters={
            "quadrature_degree": quadrature_degree})

    return dolfinx_form


def facet_area_avg_dg0(mesh, quadrature_degree=None):
    DG0 = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
    facet_area_avg = dolfinx.fem.Function(DG0)
    v_dg = ufl.TestFunction(DG0)
    num_facets = dolfinx.cpp.mesh.cell_num_entities(
        mesh.topology.cell_type, mesh.topology.dim - 1)
    avg_facet_area = ((v_dg("+") + v_dg("-"))/num_facets)*ufl.dS \
                     + (v_dg/num_facets)*ufl.ds
    dolfinx_form = _form_with_estimated_quad_degree(
        mesh, avg_facet_area, quadrature_degree)
    dolfinx.fem.assemble_vector(facet_area_avg.vector, dolfinx_form)
    facet_area_avg.vector.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return facet_area_avg


def cell_volume_dg0(mesh, quadrature_degree=None):
    DG0 = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
    cell_volume = dolfinx.fem.Function(DG0)
    v_dg = ufl.TestFunction(DG0)
    cell_volume_form = v_dg*ufl.dx
    dolfinx_form = _form_with_estimated_quad_degree(
        mesh, cell_volume_form, quadrature_degree)
    dolfinx.fem.assemble_vector(cell_volume.vector, dolfinx_form)
    cell_volume.vector.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return cell_volume
