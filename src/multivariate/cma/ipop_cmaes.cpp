/*
 Copyright (c) 2020 Mike Gimelfarb

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the > "Software"), to
 deal in the Software without restriction, including without limitation the
 rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, > subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ================================================================
 REFERENCES:

 [1] Auger, Anne, and Nikolaus Hansen. "A restart CMA evolution strategy with
 increasing population size." Evolutionary Computation, 2005. The 2005 IEEE
 Congress on. Vol. 2. IEEE, 2005.

 [2] Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box
 Optimization Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the
 BBOB-2012 Noiseless Testbed." Genetic and Evolutionary Computation Conference
 (GECCO-2012), ACM Press : 269-276. July 2012.

 [3] Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Alternative
 restart strategies for CMA-ES." International Conference on Parallel Problem
 Solving from Nature. Springer, Berlin, Heidelberg, 2012.

 [4] Liao, Tianjun, and Thomas Stützle. "Bounding the population size of
 IPOP-CMA-ES on the noiseless BBOB testbed." Proceedings of the 15th annual
 conference companion on Genetic and evolutionary computation. ACM, 2013.

 [5] https://coco.gforge.inria.fr/lib/exe/fetch.php?media=slides2013:00
 -bbob2013slides.pdf
 */

#include <cmath>

#include "../../random.hpp"

#include "ipop_cmaes.h"

using Random = effolkronium::random_static;

IPopCmaes::IPopCmaes(BaseCmaes *base, int mfev, bool print, double sigma0,
		bool nipop, double ksigmadec, bool boundlambda) {
	_base = base;
	_mfev = mfev;
	_print = print;
	_sigmadef = sigma0;
	_nipop = nipop;
	_ksigmadec = ksigmadec;
	_boundlambda = boundlambda;
}

void IPopCmaes::init(const multivariate_problem &f, const double *guess) {

	// initialize domain
	if (f._hasc || f._hasbbc) {
		std::cerr
				<< "Warning [IPOP-CMAES]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);
	_guess = std::vector<double>(guess, guess + _n);
	_fev = 0;

	// initialize lambda
	_lambda = _lambdadef = 4 + static_cast<int>(3. * std::log(1. * _n));
	_lambdamax = 10 * _n * _n;

	// initialize sigma
	_sigma = _sigmadef;

	// initialize number of function evaluations
	const int maxfev = computeMaxEvaluations(_lambdadef);

	// first default run with small population size
	_base->setParams(_lambdadef, _sigmadef, maxfev);
	const auto &sol = _base->optimize(_f, &_guess[0]);
	const auto &x = sol._sol;
	_fx = _f._f(&x[0]);
	_fev += sol._fev + 1;

	// initialize memory
	_it = 0;
	_fbest = _fx;
	_xbest = std::vector<double>(x);
	_x0 = std::vector<double>(_n);
	_table = Tabular();

	// print output
	if (_print) {
		_table.setWidth( { 5, 10, 5, 25, 25, 25 });
		_table.printRow("run", "budget", "pop", "sigma", "f*", "best f*");
		_table.printRow(_it, _fev, _lambdadef, _sigmadef, _fx, _fbest);
	}
}

void IPopCmaes::iterate() {

	// evolve the initial guess
	for (int i = 0; i < _n; i++) {
		_x0[i] = Random::get(_lower[i], _upper[i]);
	}

	// set CMA parameters
	// apply the optional lambda cycling when reaching the maximum lambda
	// also tuned the factor from 2 to 2.88
	if (_boundlambda) {
		_lambda <<= 1;
		if (_lambda > _lambdamax) {
			if (_lambda - _lambdamax < _lambdamax - (_lambda >> 1)) {
				_lambda = _lambdamax;
			} else {
				_lambda = _lambdadef;
			}
		}
	} else {
		_lambda <<= 1;
	}
	if (_nipop) {
		_sigma /= _ksigmadec;
		_sigma = std::max(_sigma, 0.01 * _sigmadef);
	}

	// maximum allowed function evaluations
	const int maxfev = computeMaxEvaluations(_lambda);

	// run CMA with increasing population size
	_base->setParams(_lambda, _sigma, maxfev);
	const auto &sol = _base->optimize(_f, &_x0[0]);
	const auto &x = sol._sol;
	_fx = _f._f(&x[0]);
	_fev += sol._fev + 1;

	// update the best point so far
	if (_fx < _fbest) {
		_fbest = _fx;
		_xbest = std::vector<double>(x);
	}

	// increment counters and adjust budget
	_it++;

	// print output
	if (_print) {
		_table.printRow(_it, _fev, _lambda, _sigma, _fx, _fbest);
	}
}

multivariate_solution IPopCmaes::solution(){
	return {_xbest, _fev, false};
}

multivariate_solution IPopCmaes::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	bool converged = false;
	while (_fev < _mfev) {
		iterate();
	}
	return {_xbest, _fev, converged};
}

int IPopCmaes::computeMaxEvaluations(const int lambda) {

	// maximum number of iterations according to MaxIter in paper
	const int maxit = static_cast<int>(100.
			+ 50. * (_n + 3) * (_n + 3) / std::sqrt(1. * lambda));

	// each iteration consists of lambda evaluations
	const int maxfev = maxit * lambda;

	// limit to the remaining budget
	return std::min(maxfev, _mfev - _fev);
}
