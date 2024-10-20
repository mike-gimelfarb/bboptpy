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

 [1] Hansen, Nikolaus. "Benchmarking a BI-population CMA-ES on the BBOB-2009
 function testbed." Proceedings of the 11th Annual Conference Companion on
 Genetic and Evolutionary Computation Conference: Late Breaking Papers. ACM,
 2009.

 [2] Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box
 Optimization Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the
 BBOB-2012 Noiseless Testbed." Genetic and Evolutionary Computation Conference
 (GECCO-2012), ACM Press : 269-276. July 2012.

 [3] Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Alternative
 restart strategies for CMA-ES." International Conference on Parallel Problem
 Solving from Nature. Springer, Berlin, Heidelberg, 2012.
 */

#include <cmath>
#include <iostream>

#include "../../random.hpp"

#include "bipop_cmaes.h"

using Random = effolkronium::random_static;

BiPopCmaes::BiPopCmaes(BaseCmaes *base, int mfev, bool print, double sigma0,
		int maxlargeruns, bool nbipop, double ksigmadec, double kbudget) {
	_sigmadef = sigma0;
	_mfev = mfev;
	_print = print;
	_base = base;
	_maxlargeruns = maxlargeruns;
	_nbipop = nbipop;
	_ksigmadec = ksigmadec;
	_kbudget = kbudget;
}

void BiPopCmaes::init(const multivariate_problem &f, const double *guess) {

	// initialize problem
	if (f._hasc || f._hasbbc) {
		std::cerr
				<< "Warning [BIPOP-CMAES]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);
	_guess = std::vector<double>(guess, guess + _n);
	_fev = 0;

	// initialize lambda
	_lambdadef = 4 + static_cast<int>(3. * std::log(1. * _n));

	// maximum allowed function evaluations
	const int maxfev = computeMaxEvaluations(_lambdadef);

	// first default run with small population size
	_base->setParams(_lambdadef, _sigmadef, maxfev);
	const auto &sol = _base->optimize(_f, &_guess[0]);
	const auto &x = sol._sol;
	_fx = _f._f(&x[0]);
	_fev += sol._fev + 1;

	// initialize memory
	_largebudget = _smallbudget = 0;
	_largerestarts = _smallrestarts = 0;
	_bestregime = 1;
	_it = 0;
	_fxbest = _fx;
	_xbest = std::vector<double>(x);
	_x0 = std::vector<double>(_n);
	_table = Tabular();

	// print output
	if (_print) {
		_table.setWidth( { 5, 5, 5, 5, 10, 10, 10, 5, 25, 25, 25 });
		_table.printRow("run", "regime", "run1", "run2", "budget1", "budget2",
				"fev", "pop", "sigma", "f*", "best f*");
		_table.printRow(_it, 0, _largerestarts, _smallrestarts, _largebudget,
				_smallbudget, _fev, _lambdadef, _sigmadef, _fx, _fxbest);
	}
}

void BiPopCmaes::iterate() {

	// evolve the initial guess
	for (int i = 0; i < _n; i++) {
		_x0[i] = Random::get(_lower[i], _upper[i]);
	}

	// decide which regime to run
	int regime;
	if (_nbipop) {

		// allocate more budget to the better-performing regime (NBIPOP)
		if (_bestregime == 1) {
			if (_largebudget <= _smallbudget * _kbudget) {
				regime = 1;
			} else {
				regime = 2;
			}
		} else {
			if (_smallbudget <= _kbudget * _largebudget) {
				regime = 2;
			} else {
				regime = 1;
			}
		}
	} else {

		// allocate equal budget to both regimes (BIPOP)
		if (_largebudget <= _smallbudget) {
			regime = 1;
		} else {
			regime = 2;
		}
	}

	// apply the regime for one iteration
	if (regime == 1) {
		runFirstRegime();
	} else {
		runSecondRegime();
	}
	_it++;

	// print output
	if (_print) {
		if (regime == 1) {
			_table.printRow(_it, regime, _largerestarts, _smallrestarts,
					_largebudget, _smallbudget, _fev, _largelambda, _largesigma,
					_fx, _fxbest);
		} else {
			_table.printRow(_it, regime, _largerestarts, _smallrestarts,
					_largebudget, _smallbudget, _fev, _smalllambda, _smallsigma,
					_fx, _fxbest);
		}
	}
}

multivariate_solution BiPopCmaes::solution(){
	return {_xbest, _fev, false};
}

multivariate_solution BiPopCmaes::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	bool converged = false;
	while (true) {
		iterate();
		if (_largerestarts >= _maxlargeruns) {
			if (_print) {
				std::cerr
						<< "Warning [BIPOP-CMAES]: reached the maximum number of large population restarts."
						<< std::endl;
			}
			break;
		}
		if (_fev >= _mfev) {
			break;
		}
	}
	return {_xbest, _fev, converged};
}

int BiPopCmaes::computeMaxEvaluations(const int lambda) {

	// maximum number of iterations according to MaxIter in paper
	const int maxit = static_cast<int>(100.
			+ 50. * (_n + 3) * (_n + 3) / std::sqrt(1. * lambda));

	// each iteration consists of lambda evaluations
	const int maxfev = maxit * lambda;

	// limit to the remaining budget
	return std::min(maxfev, _mfev - _fev);
}

void BiPopCmaes::runFirstRegime() {

	// set CMA parameters
	_largelambda =
			static_cast<int>(_lambdadef * std::pow(2, _largerestarts + 1));
	if (_nbipop) {
		_largesigma = _sigmadef * std::pow(1. / _ksigmadec, _largerestarts + 1);
		_largesigma = std::max(_largesigma, 0.01 * _sigmadef);
	} else {
		_largesigma = _sigmadef;
	}

	// maximum allowed function evaluations
	const int maxfev = computeMaxEvaluations(_largelambda);

	// run CMA with increasing population size
	_base->setParams(_largelambda, _largesigma, maxfev);
	const auto &sol = _base->optimize(_f, &_x0[0]);
	const auto &x = sol._sol;
	_fx = _f._f(&x[0]);
	_fev += sol._fev + 1;

	// update the best point so far
	if (_fx < _fxbest) {
		_fxbest = _fx;
		_xbest = std::vector<double>(x);
		_bestregime = 1;
	}

	// increment counters and adjust budget
	_largebudget += sol._fev;
	_largerestarts++;
}

void BiPopCmaes::runSecondRegime() {

	// set CMA parameters
	const double u = Random::get(0., 1.);
	_smalllambda = static_cast<int>(_lambdadef
			* std::pow((0.5 * _largelambda) / _lambdadef, u * u));
	_smallsigma = _sigmadef * std::pow(10., -2. * Random::get(0., 1.));

	// the paper recommends to enforce a small max fev of 1/2 the large budget
	int maxfev = computeMaxEvaluations(_smalllambda);
	maxfev = std::min(maxfev, _largebudget >> 1);

	// run CMA with small population size
	_base->setParams(_smalllambda, _smallsigma, maxfev);
	const auto &sol = _base->optimize(_f, &_x0[0]);
	const auto &x = sol._sol;
	_fx = _f._f(&x[0]);
	_fev += sol._fev + 1;

	// update the best point so far
	if (_fx < _fxbest) {
		_fxbest = _fx;
		_xbest = std::vector<double>(x);
		_bestregime = 2;
	}

	// increment counters and adjust budget
	_smallbudget += sol._fev;
	_smallrestarts++;
}
