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

 [1] Lasdon, Leon, et al. "Adaptive memory programming for constrained global
 optimization." Computers & Operations Research 37.8 (2010): 1500-1509.

 [2] The AMPGO Solver — AMPGO 0.1.0 documentation. (n.d.). Infinity77.Net.
 Retrieved August 15, 2021, from http://infinity77.net/global_optimization/ampgo.html
 */

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <numeric>

#include "../../blas.h"
#include "../../random.hpp"

#include "ampttm.h"

using Random = effolkronium::random_static;

AMPTTM::AMPTTM(MultivariateOptimizer *local, int mfev, bool print, double eps1,
		double eps2, int totaliter, int maxiter, unsigned int tabutenure,
		tabu_removal_strategy remove) {
	_local = local;
	_mfev = mfev;
	_print = print;
	_eps1 = eps1;
	_eps2 = eps2;
	_tit = totaliter;
	_mit = maxiter;
	_tabutenure = tabutenure;
	_remove = remove;
}

void AMPTTM::init(const multivariate_problem &f, const double *guess) {

	// define problem
	if (f._hasc) {
		std::cerr
				<< "Warning [AMPTTM]: equality and inequality constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_guess = std::vector<double>(guess, guess + _n);
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// memory
	_fev = _gev = 0;
	_it = 0;
	_temp = std::vector<double>(_n);
	_s = std::vector<double>(_guess);
	_bestf = std::numeric_limits<double>::infinity();
	_bestx = std::vector<double>(_n);
	_bestxfeas = false;
	_x0 = std::vector<double>(_n);
	_tabu = std::vector<std::vector<double>>();
	_table = Tabular();
}

void AMPTTM::iterate() {

	// minimization of original objective
	const auto &sol = solveLocalProblem(&_s[0]);
	std::vector<double> s = sol._sol;

	// tabu tunneling
	const double fold = _bestf;
	int tunnels = 0;
	_improve = false;
	while (tunnels < _mit && !_improve && _fev < _mfev && _gev < _mfev) {
		const auto &sol1 = solveProjectionProblem(&s[0]);
		const auto &sp = sol1._sol;
		const auto &sol2 = solveTunnelingProblem(&sp[0]);
		const auto &spp = sol2._sol;
		std::copy(spp.begin(), spp.end(), s.begin());
		tunnels++;
	}

	// print
	if (_print) {
		const double fs = _f._f(&s[0]);
		_fev++;
		bool feas;
		if (_f._hasbbc) {
			feas = _f._bbc(&s[0]);
			_gev++;
		} else {
			feas = true;
		}
		_table.printRow(_it, _fev, fs, feas, _bestf, _bestxfeas, tunnels,
				_bestf < fold);
	}

	// the final point is used as the initial guess for the next iteration
	std::copy(s.begin(), s.end(), _s.begin());
	_it++;
}

multivariate_solution AMPTTM::solution(){
	return {_bestx, _fev, 0, _gev, false};
}

multivariate_solution AMPTTM::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);

	// print header
	if (_print) {
		_table.setWidth( { 5, 10, 25, 5, 25, 5, 7, 6 });
		_table.printRow("iter", "fev", "f*", "feas", "best f*", "feas",
				"tunnels", "success");
	}

	// main loop
	while (_it < _tit && _fev < _mfev && _gev < _mfev) {
		iterate();
	}
	return {_bestx, _fev, 0, _gev, false};
}

void AMPTTM::updateTabu(const double *x) {
	while (_tabu.size() >= _tabutenure) {
		switch (_remove) {
		case farthest: {

			// delete element farthest from x
			double bdist = std::numeric_limits<double>::infinity();
			int bi = -1;
			int i = 0;
			for (const auto &tbp : _tabu) {
				for (int j = 0; j < _n; j++) {
					_temp[j] = tbp[j] - x[j];
				}
				const double dist = std::inner_product(_temp.begin(),
						_temp.end(), _temp.begin(), 0.);
				if (dist < bdist) {
					bdist = dist;
					bi = i;
				}
				i++;
			}
			if (bi >= 0) {
				_tabu.erase(_tabu.begin() + bi);
			}
			break;
		}
		default: {

			// delete oldest element
			_tabu.erase(_tabu.begin() + 0);
			break;
		}
		}
	}

	// append the current x
	const std::vector<double> v(x, x + _n);
	_tabu.push_back(v);
}

multivariate_solution AMPTTM::solveLocalProblem(const double *guess) {

	// call local solver
	const auto &sol = _local->optimize(_f, guess);
	const auto &x = sol._sol;
	_fev += sol._fev;
	_gev += sol._bbev;

	// update the tabu list
	updateTabu(&x[0]);

	// update best fitness
	const double fs = _f._f(&x[0]);
	_fev += 1;
	if (fs < _bestf) {
		_bestf = fs;
		std::copy(x.begin(), x.end(), _bestx.begin());
		if (_f._hasbbc) {
			_bestxfeas = _f._bbc(&x[0]);
			_gev += 1;
		} else {
			_bestxfeas = true;
		}
	}

	return sol;
}

multivariate_solution AMPTTM::solveProjectionProblem(const double *guess) {

	// construct the guess by tunneling
	for (int i = 0; i < _n; i++) {
		_temp[i] = Random::get(-1., 1.);
	}
	const double snorm = dnrm2(_n, guess);
	const double rnorm = dnrm2(_n, &_temp[0]);
	double beta = _eps2 * snorm / rnorm;
	if (beta < 1e-8) {
		beta = _eps2;
	}
	for (int i = 0; i < _n; i++) {
		_x0[i] = guess[i] + beta * _temp[i];
	}

	// ensure the guess lies in the bounds
	for (int i = 0; i < _n; i++) {
		_x0[i] = std::max(_lower[i], std::min(_x0[i], _upper[i]));
	}

	// solve the projection subproblem
	if (_f._hasbbc) {

		// minimize ||x - x0|| subject to the constraints
		const multivariate &ttf = [&](const double *x) -> double {
			for (int i = 0; i < _n; i++) {
				_temp[i] = x[i] - _x0[i];
			}
			return std::inner_product(_temp.begin(), _temp.end(), _temp.begin(),
					0.);
		};
		const multivariate_problem ttfp { ttf, _n, &_lower[0], &_upper[0],
				_f._bbc };

		// solve the projection subproblem
		const auto &sol = _local->optimize(ttfp, &_x0[0]);
		const auto &x = sol._sol;

		// check for feasibility
		if (!_f._bbc(&x[0])) {
			std::cerr
					<< "Warning [AMPTTM]: projection subproblem returned an infeasible solution."
					<< std::endl;
		}
		_gev++;
		return sol;
	} else {
		return {_x0, 0, true};
	}
}

multivariate_solution AMPTTM::solveTunnelingProblem(const double *guess) {

	// compute aspiration value
	const double aspiration = _bestf - _eps1 * (1. + std::fabs(_bestf));

	// define the tunneling subproblem
	const multivariate &ttf = [&](const double *x) -> double {
		const double fx = _f._f(x);
		const double improvement = std::pow(fx - aspiration, 2.);
		double tabu_penalty = 1.;
		for (const auto &tbp : _tabu) {
			for (int i = 0; i < _n; i++) {
				_temp[i] = x[i] - tbp[i];
			}
			const double dnorm2 = std::inner_product(_temp.begin(), _temp.end(),
					_temp.begin(), 0.);
			tabu_penalty *= dnorm2;
		}
		return improvement / tabu_penalty;
	};
	multivariate_problem problem;
	if (_f._hasbbc) {
		problem = multivariate_problem { ttf, _n, &_lower[0], &_upper[0],
				_f._bbc };
	} else {
		problem = multivariate_problem { ttf, _n, &_lower[0], &_upper[0] };
	}

	// solve the tunneling subproblem
	const auto &sol = _local->optimize(problem, guess);
	const auto &x = sol._sol;
	_fev += sol._fev;
	_gev += sol._bbev;

	// update the tabu list
	updateTabu(&x[0]);

	// update best fitness
	const double fs = _f._f(&x[0]);
	_fev += 1;
	if (fs < _bestf) {
		_bestf = fs;
		std::copy(x.begin(), x.end(), _bestx.begin());
		_improve = true;
		if (_f._hasbbc) {
			_bestxfeas = _f._bbc(&x[0]);
			_gev += 1;
		} else {
			_bestxfeas = true;
		}
	}

	return sol;
}
