#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "pybind11/numpy.h"
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "../src/multivariate/multivariate.h"
#include "../src/multivariate/acd/acd.h"
#include "../src/multivariate/amalgam/amalgam.h"
#include "../src/multivariate/basin/basinhopping.h"
#include "../src/multivariate/cma/base_cmaes.h"
#include "../src/multivariate/cma/cmaes.h"
#include "../src/multivariate/cma/active_cmaes.h"
#include "../src/multivariate/cma/cholesky_cmaes.h"
#include "../src/multivariate/cma/lm_cmaes.h"
#include "../src/multivariate/cma/sep_cmaes.h"
#include "../src/multivariate/cma/ipop_cmaes.h"
#include "../src/multivariate/cma/bipop_cmaes.h"
#include "../src/multivariate/nes/xnes.h"
#include "../src/multivariate/de/jade.h"
#include "../src/multivariate/de/shade.h"
#include "../src/multivariate/de/sansde.h"
#include "../src/multivariate/de/ssde.h"
#include "../src/multivariate/jaya/jaya.h"
#include "../src/multivariate/direct/directl.h"
#include "../src/multivariate/harmony/nshs.h"
#include "../src/multivariate/hees/hees.h"
#include "../src/multivariate/powell/bobyqa.h"
#include "../src/multivariate/powell/newuoa.h"
#include "../src/multivariate/praxis/praxis.h"
#include "../src/multivariate/pso/apso.h"
#include "../src/multivariate/pso/ccpso.h"
#include "../src/multivariate/pso/cso.h"
#include "../src/multivariate/pso/ds.h"
#include "../src/multivariate/pso/slpso.h"
#include "../src/multivariate/rosenbrock/rosenbrock.h"
#include "../src/multivariate/simplex/nelder_mead.h"
#include "../src/multivariate/crs/crs.h"

namespace py = pybind11;
using namespace pybind11::literals;

void build_acd(py::module_ &m) {
	py::class_<ACD, MultivariateOptimizer> solver(m, "ACD");
	solver.def(py::init<int, double, double, double, double>(), "mfev"_a,
			"ftol"_a, "xtol"_a, "ksucc"_a = 2., "kunsucc"_a = 0.5);
}

//void build_algencan(py::module_ &m) {
//	py::class_<Algencan, MultivariateOptimizer> solver(m, "ALGENCAN");
//	solver.def(
//			py::init<MultivariateOptimizer*, int, double, bool, double, double,
//					double, double>(), "local"_a, "mit"_a, "tol"_a, "print"_a =
//					false, "tau"_a = 0.5, "gamma"_a = 10., "lambda0"_a = 0.,
//			"mu0"_a = 0.);
//}

void build_amalgam(py::module_ &m) {
	py::class_<Amalgam, MultivariateOptimizer> solver(m, "AMALGAM");
	solver.def(py::init<int, double, double, int, bool, bool, bool>(), "mfev"_a,
			"tol"_a, "stol"_a, "np"_a = 0, "iamalgam"_a = true, "noparam"_a =
					true, "print"_a = true);
}

//void build_amp(py::module_ &m) {
//	py::class_<AMPTTM, MultivariateOptimizer> solver(m, "AMP");
//
//	py::enum_<AMPTTM::tabu_removal_strategy> tabu_removal(solver,
//			"AMP_TabuRemoveStrategy");
//	tabu_removal.value("oldest", AMPTTM::tabu_removal_strategy::oldest);
//	tabu_removal.value("farthest", AMPTTM::tabu_removal_strategy::farthest);
//	tabu_removal.export_values();
//
//	solver.def(
//			py::init<MultivariateOptimizer*, int, bool, double, double, int,
//					int, unsigned int, AMPTTM::tabu_removal_strategy>(),
//			"local"_a, "mfev"_a, "print"_a = false, "eps1"_a = 0.02, "eps2"_a =
//					0.1, "totaliter"_a = 9999, "maxiter"_a = 5, "tabutenure"_a =
//					5, "remove"_a = AMPTTM::tabu_removal_strategy::farthest);
//}

void build_basin_hopping(py::module_ &m) {
	py::class_<StepsizeStrategy> step(m, "BasinHopping_StepStrategy");
	step.def(py::init<double>(), "stepsize"_a);

	py::class_<AdaptiveStepsizeStrategy, StepsizeStrategy> adaptivestep(m,
			"BasinHopping_AdaptStrategy");
	adaptivestep.def(py::init<double, double, int, double>(), "stepsize"_a = 1.,
			"accept_rate"_a = 0.5, "interval"_a = 5, "factor"_a = 0.9);

	py::class_<BasinHopping, MultivariateOptimizer> solver(m, "BasinHopping");
	solver.def(
			py::init<MultivariateOptimizer*, StepsizeStrategy*, bool, int,
					double>(), "minimizer"_a, "stepstrat"_a, "print"_a = true,
			"mit"_a = 99, "temp"_a = 1.);
}

void build_base_cmaes(py::module_ &m) {
	py::class_<BaseCmaes, MultivariateOptimizer> solver(m, "BaseCMAES");
}

void build_cmaes(py::module_ &m) {
	py::class_<Cmaes, BaseCmaes> solver(m, "CMAES");
	solver.def(py::init<int, double, int, double, bool, double>(), "mfev"_a,
			"tol"_a, "np"_a, "sigma0"_a = 2., "bound"_a = false, "eigenrate"_a =
					0.25);
}

void build_active_cmaes(py::module_ &m) {
	py::class_<ActiveCmaes, Cmaes> solver(m, "ActiveCMAES");
	solver.def(py::init<int, double, int, double, bool, double, double>(),
			"mfev"_a, "tol"_a, "np"_a, "sigma0"_a = 2., "bound"_a = false,
			"alphacov"_a = 2., "eigenrate"_a = 0.25);
}

void build_cholesky_cmaes(py::module_ &m) {
	py::class_<CholeskyCmaes, BaseCmaes> solver(m, "CholeskyCMAES");
	solver.def(py::init<int, double, double, int, double, bool>(), "mfev"_a,
			"tol"_a, "stol"_a, "np"_a, "sigma0"_a = 2., "bound"_a = false);
}

void build_lm_cmaes(py::module_ &m) {
	py::class_<LmCmaes, BaseCmaes> solver(m, "LmCMAES");
	solver.def(py::init<int, double, int, int, double, bool, bool, bool>(),
			"mfev"_a, "tol"_a, "np"_a, "memory"_a = 0, "sigma0"_a = 2.,
			"bound"_a = false, "rademacher"_a = true, "usenew"_a = true);
}

void build_sep_cmaes(py::module_ &m) {
	py::class_<SepCmaes, BaseCmaes> solver(m, "SepCMAES");
	solver.def(py::init<int, double, int, double, bool, bool>(), "mfev"_a,
			"tol"_a, "np"_a, "sigma0"_a = 2., "bound"_a = false, "adjustlr"_a =
					true);
}

void build_ipop_cmaes(py::module_ &m) {
	py::class_<IPopCmaes, MultivariateOptimizer> solver(m, "IPopCMAES");
	solver.def(py::init<BaseCmaes*, int, bool, double, bool, double, bool>(),
			"base"_a, "mfev"_a, "print"_a = false, "sigma0"_a = 2., "nipop"_a =
					true, "ksigmadec"_a = 1.6, "boundlambda"_a = true);
}

void build_bipop_cmaes(py::module_ &m) {
	py::class_<BiPopCmaes, MultivariateOptimizer> solver(m, "BiPopCMAES");
	solver.def(
			py::init<BaseCmaes*, int, bool, double, int, bool, double, double>(),
			"base"_a, "mfev"_a, "print"_a = false, "sigma0"_a = 2.,
			"maxlargeruns"_a = 9, "nbipop"_a = true, "ksigmadec"_a = 1.6,
			"kbudget"_a = 2.);
}

void build_xnes(py::module_ &m) {
	py::class_<xNES, MultivariateOptimizer> solver(m, "xNES");
	solver.def(py::init<int, double, double, double>(), "mfev"_a, "tol"_a,
			"a0"_a = 1.0, "etamu"_a = 1.0);
}

void build_jade(py::module_ &m) {
	py::class_<JadeSearch, MultivariateOptimizer> solver(m, "JADE");
	solver.def(py::init<int, int, double, bool, bool, double, double, double>(),
			"mfev"_a, "np"_a, "tol"_a, "archive"_a = true, "repaircr"_a = true,
			"pelite"_a = 0.05, "cdamp"_a = 0.1, "sigma"_a = 0.07);
}

void build_shade(py::module_ &m) {
	py::class_<ShadeSearch, MultivariateOptimizer> solver(m, "SHADE");
	solver.def(py::init<int, int, double, bool, bool, int, int>(), "mfev"_a,
			"npinit"_a, "tol"_a, "archive"_a = true, "repaircr"_a = true,
			"h"_a = 100, "npmin"_a = 4);
}

void build_sansde(py::module_ &m) {
	py::class_<SaNSDESearch, MultivariateOptimizer> solver(m, "SANSDE");
	solver.def(py::init<int, int, double, bool, int, int, int>(), "mfev"_a,
			"np"_a, "tol"_a, "repaircr"_a = true, "crref"_a = 5, "pupdate"_a =
					50, "crupdate"_a = 25);
}

void build_ssde(py::module_ &m) {
	py::class_<SSDESearch, MultivariateOptimizer> solver(m, "SSDE");
	solver.def(py::init<int, int, double, int, int, double, int, bool, bool>(),
			"mfev"_a, "npinit"_a, "tol"_a, "patience"_a = 1000, "npmin"_a = 4,
			"ptop"_a = 0.11, "h"_a = 100, "usede"_a = false, "repaircr"_a =
					true);
}

void build_ds(py::module_ &m) {
	py::class_<DSSearch, MultivariateOptimizer> solver(m, "DSA");
	solver.def(py::init<int, double, double, int, bool, int>(), "mfev"_a,
			"tol"_a, "stol"_a, "np"_a, "adapt"_a = true, "nbatch"_a = 100);
}

void build_directl(py::module_ &m) {
	py::class_<Directl, MultivariateOptimizer> solver(m, "DIRECT");
	solver.def(py::init<int, double, double, double, int>(), "mfev"_a,
			"volper"_a, "sigmaper"_a, "eps"_a = 0., "method"_a = 0);
}

void build_nshs(py::module_ &m) {
	py::class_<NSHS, MultivariateOptimizer> solver(m, "NSHS");
	solver.def(py::init<int, int, double>(), "mfev"_a, "hms"_a, "fstdmin"_a =
			0.0001);
}

void build_hees(py::module_ &m) {
	py::class_<Hees, MultivariateOptimizer> solver(m, "HEES");
	solver.def(py::init<int, double, int, bool, int, double>(), "mfev"_a,
			"tol"_a, "mres"_a = 1, "print"_a = false, "np"_a = 0, "sigma0"_a =
					2.);
}

void build_jaya(py::module_ &m) {
	py::class_<JayaSearch, MultivariateOptimizer> solver(m, "JAYA");

	py::enum_<JayaSearch::jaya_mutation_method> mutation_strategy(solver,
			"JAYA_Mutation");
	mutation_strategy.value("original",
			JayaSearch::jaya_mutation_method::original);
	mutation_strategy.value("levy", JayaSearch::jaya_mutation_method::levy);
	mutation_strategy.value("tent_map",
			JayaSearch::jaya_mutation_method::tent_map);
	mutation_strategy.value("logistic",
			JayaSearch::jaya_mutation_method::logistic);
	mutation_strategy.export_values();

	solver.def(
			py::init<int, double, int, int, bool, int,
					JayaSearch::jaya_mutation_method, double, double, int,
					double>(), "mfev"_a, "tol"_a, "np"_a, "npmin"_a, "adapt"_a =
					true, "k0"_a = 2, "mutation"_a =
					JayaSearch::jaya_mutation_method::logistic, "scale"_a =
					0.01, "beta"_a = 1.5, "kcheb"_a = 2, "temper"_a = 10.);
}

//void build_mayfly(py::module_ &m) {
//	py::class_<MayflySearch, MultivariateOptimizer> solver(m, "Mayfly");
//	solver.def(
//			py::init<int, int, double, double, double, double, double, double,
//					double, double, double, double, double, double, double,
//					double, double, bool>(), "np"_a, "mfev"_a, "a1"_a = 1.,
//			"a2"_a = 1.5, "a3"_a = 1.5, "beta"_a = 2., "dance"_a = 5.,
//			"ddamp"_a = 0.8, "fl"_a = 1., "fldamp"_a = 0.99, "gmin"_a = 0.8,
//			"gmax"_a = 0.8, "vdamp"_a = 0.1, "sigma"_a = 0.1, "pmutdim"_a =
//					0.01, "pmutnp"_a = 0.05, "l"_a = 0.95, "pgb"_a = false);
//}

void build_bobyqa(py::module_ &m) {
	py::class_<Bobyqa, MultivariateOptimizer> solver(m, "BOBYQA");
	solver.def(py::init<int, int, double, double>(), "mfev"_a, "np"_a, "rho"_a,
			"tol"_a);
}

void build_newuoa(py::module_ &m) {
	py::class_<Newuoa, MultivariateOptimizer> solver(m, "NEWUOA");
	solver.def(py::init<int, int, double, double>(), "mfev"_a, "np"_a, "rho"_a,
			"tol"_a);
}

void build_praxis(py::module_ &m) {
	py::class_<Praxis, MultivariateOptimizer> solver(m, "PRAXIS");
	solver.def(py::init<double, double>(), "tol"_a, "mstep"_a);
}

void build_apso(py::module_ &m) {
	py::class_<APSOSearch, MultivariateOptimizer> solver(m, "APSO");
	solver.def(py::init<int, double, int, bool>(), "mfev"_a, "tol"_a, "np"_a,
			"correct"_a = true);
}

void build_cso(py::module_ &m) {
	py::class_<CSOSearch, MultivariateOptimizer> solver(m, "CSO");
	solver.def(py::init<int, double, int, int, bool, bool, double>(), "mfev"_a,
			"stol"_a, "np"_a, "pcompete"_a = 3, "ring"_a = false, "correct"_a =
					true, "vmax"_a = 0.2);
}

CCPSOSearch init_ccpso(int mfev, double sigmatol, int np, std::vector<int> &pps,
		int npps, bool correct, double pcauchy, MultivariateOptimizer *local,
		int localfreq) {
	return CCPSOSearch(mfev, sigmatol, np, &pps[0], npps, correct, pcauchy,
			local, localfreq);
}

void build_ccpso(py::module_ &m) {
	py::class_<CCPSOSearch, MultivariateOptimizer> solver(m, "CCPSO");
	solver.def(py::init(&init_ccpso), "mfev"_a, "sigmatol"_a, "np"_a, "pps"_a,
			"npps"_a, "correct"_a = true, "pcauchy"_a = -1., "local"_a =
					nullptr, "localfreq"_a = 10);
}

void build_slpso(py::module_ &m) {
	py::class_<SLPSOSearch, MultivariateOptimizer> solver(m, "SLPSO");
	solver.def(
			py::init<int, double, int, double, double, double, double, double,
					double>(), "mfev"_a, "stol"_a, "np"_a, "omegamin"_a = 0.4,
			"omegamax"_a = 0.9, "eta"_a = 1.496, "gamma"_a = 0.01, "vmax"_a =
					0.2, "Ufmax"_a = 10.);
}

void build_rosenbrock(py::module_ &m) {
	py::class_<Rosenbrock, MultivariateOptimizer> solver(m, "Rosenbrock");
	solver.def(py::init<int, double, double, double>(), "mfev"_a, "tol"_a,
			"step0"_a, "decf"_a = 0.1);
}

void build_simplex(py::module_ &m) {
	py::class_<NelderMead, MultivariateOptimizer> solver(m, "NelderMead");

	py::enum_<NelderMead::simplex_initializer> simplex_init(solver,
			"NelderMead_SimplexInit");
	simplex_init.value("coordinate_axis",
			NelderMead::simplex_initializer::coordinate_axis);
	simplex_init.value("spendley", NelderMead::simplex_initializer::spendley);
	simplex_init.value("pfeffer", NelderMead::simplex_initializer::pfeffer);
	simplex_init.value("random", NelderMead::simplex_initializer::random);
	simplex_init.export_values();

	py::enum_<NelderMead::parameter_initializer> parameter_init(solver,
			"NelderMead_ParamInit");
	parameter_init.value("original",
			NelderMead::parameter_initializer::original);
	parameter_init.value("gao2010", NelderMead::parameter_initializer::gao2010);
	parameter_init.value("mehta2019_crude",
			NelderMead::parameter_initializer::mehta2019_crude);
	parameter_init.value("mehta2019_refined",
			NelderMead::parameter_initializer::mehta2019_refined);
	parameter_init.export_values();

	solver.def(
			py::init<int, double, double, NelderMead::simplex_initializer,
					NelderMead::parameter_initializer, int, double>(), "mfev"_a,
			"tol"_a, "rad0"_a, "minit"_a =
					NelderMead::simplex_initializer::spendley, "pinit"_a =
					NelderMead::parameter_initializer::mehta2019_refined,
			"checkev"_a = 10, "eps"_a = 1e-3);
}

void build_crs(py::module_ &m) {
	py::class_<CrsSearch, MultivariateOptimizer> solver(m, "CRS");
	solver.def(py::init<int, int, double>(), "mfev"_a, "np"_a, "tol"_a);
}

// wrap the function expressions
typedef std::function<double(const py::array_t<double>&)> py_multivariate;

// wrap the multivariable optimizer
void build_multivariate(py::module_ &m) {

	// wrap the solution object
	py::class_<multivariate_solution> solution(m, "MultivariateSolution");
	solution.def("__str__", &multivariate_solution::toString);
	solution.def_property_readonly("x", [](multivariate_solution &self) {
		return py::array_t<double>(self._sol.size(), &self._sol[0]);
	});
	solution.def_property_readonly("converged",
			[](multivariate_solution &self) {
				return self._converged;
			});
	solution.def_property_readonly("n_evals", [](multivariate_solution &self) {
		return self._fev;
	});

	// wrap the solver
	py::class_<MultivariateOptimizer> solver(m, "MultivariateSearch");

	solver.def("optimize",
			[](MultivariateOptimizer &self, py_multivariate &py_f,
					py::array_t<double> &py_lower,
					py::array_t<double> &py_upper,
					py::array_t<double> &py_guess) {
				const int n = py_lower.size();
				double *lower = static_cast<double*>(py_lower.request().ptr);
				double *upper = static_cast<double*>(py_upper.request().ptr);

				multivariate f = [py_f, n](const double *x) -> double {
					const auto &py_x = py::array_t<double>(n, x);
					return py_f(py_x);
				};

				multivariate_problem prob { f, n, lower, upper };
				double *guess = static_cast<double*>(py_guess.request().ptr);
				return self.optimize(prob, guess);
			}, "f"_a, "lower"_a, "upper"_a, "guess"_a,
			py::call_guard<py::scoped_ostream_redirect,
					py::scoped_estream_redirect>());

	solver.def("initialize",
			[](MultivariateOptimizer &self, py_multivariate &py_f,
					py::array_t<double> &py_lower,
					py::array_t<double> &py_upper,
					py::array_t<double> &py_guess) {
				const int n = py_lower.size();
				double *lower = static_cast<double*>(py_lower.request().ptr);
				double *upper = static_cast<double*>(py_upper.request().ptr);

				multivariate f = [py_f, n](const double *x) -> double {
					const auto py_x = py::array_t<double>(n, x);
					return py_f(py_x);
				};

				multivariate_problem prob { f, n, lower, upper };
				double *guess = static_cast<double*>(py_guess.request().ptr);
				return self.init(prob, guess);
			}, "f"_a, "lower"_a, "upper"_a, "guess"_a,
			py::call_guard<py::scoped_ostream_redirect,
					py::scoped_estream_redirect>());

	solver.def("iterate", &MultivariateOptimizer::iterate);

	solver.def("solution", &MultivariateOptimizer::solution);

	// put algorithm-specific bindings here
	build_acd(m);
	build_amalgam(m);
	build_basin_hopping(m);
	build_base_cmaes(m);
	build_cmaes(m);
	build_active_cmaes(m);
	build_cholesky_cmaes(m);
	build_lm_cmaes(m);
	build_sep_cmaes(m);
	build_ipop_cmaes(m);
	build_bipop_cmaes(m);
	build_xnes(m);
	build_jade(m);
	build_shade(m);
	build_sansde(m);
	build_ssde(m);
	build_ds(m);
	build_jaya(m);
	build_nshs(m);
	build_hees(m);
	build_bobyqa(m);
	build_newuoa(m);
	build_praxis(m);
	build_apso(m);
	build_cso(m);
	build_ccpso(m);
	build_slpso(m);
	build_rosenbrock(m);
	build_simplex(m);
	build_crs(m);
}
