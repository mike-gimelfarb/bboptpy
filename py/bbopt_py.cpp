/*
 * TODO: fix
 * MADS
 * MLSL
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

void build_univariate(py::module_&);
void build_multivariate(py::module_&);
// void build_constrained(py::module_&);
// void build_structured_constrained(py::module_&);

PYBIND11_MODULE(_bboptpy, m) {
	build_univariate(m);
	build_multivariate(m);
//	build_constrained(m);
//	build_structured_constrained(m);
}
