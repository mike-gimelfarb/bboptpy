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

 [1] Rao, R. "Jaya: A simple and new optimization algorithm for solving
 constrained and unconstrained optimization problems." International Journal
 of Industrial Engineering Computations 7.1 (2016): 19-34.

 [2] Rao, R. Venkata, and Ankit Saroj. "A self-adaptive multi-population based
 Jaya algorithm for engineering optimization." Swarm and Evolutionary computation
 37 (2017): 1-26.

 [3] Yu, Jiang-Tao, et al. "Jaya algorithm with self-adaptive multi-population
 and Lévy flights for solving economic load dispatch problems." IEEE Access 7
 (2019): 21372-21384.

 [4] Ravipudi, Jaya Lakshmi, and Mary Neebha. "Synthesis of linear antenna arrays
 using jaya, self-adaptive jaya and chaotic jaya algorithms." AEU-International
 Journal of Electronics and Communications 92 (2018): 54-63.
 */

#define _USE_MATH_DEFINES
#include <algorithm>
#include <math.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "../../blas.h"
#include "../../random.hpp"

#include "jaya.h"

using Random = effolkronium::random_static;

JayaSearch::JayaSearch(int mfev, double tol, int np, int npmin, bool adapt,
		int k0, jaya_mutation_method mutation, double scale, double beta,
		int kcheb, double temper) {
	_mfev = mfev;
	_stol = tol;
	_np = np;
	_adapt_k = adapt;
	_k0 = k0;
	_mutation = mutation;
	_npmin = npmin;
	_scale = scale;
	_beta = beta;
	_kcheb = kcheb;
	_temper = temper;
}

void JayaSearch::init(const multivariate_problem &f, const double *guess) {

	// define problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [JAYA]: problem constraints will be ignored."
				<< std::endl;
	}
	_n = f._n;
	_f = f;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// define parameters
	_sigmau = pow(
			(tgamma(1. + _beta) * sin(_beta * M_PI / 2.))
					/ (tgamma((1. + _beta) / 2.) * _beta
							* pow(2., (_beta - 1.) / 2.)), 1. / _beta);
	_k = _k0;
	_tmp = std::vector<double>(_n);
	_len = std::vector<int>(_np);

	// define swarm
	_xchaos = Random::get(0., 1.);
	_fgbest = std::numeric_limits<double>::infinity();
	_bestx = std::vector<double>(_n);
	_pool.clear();
	for (int i = 0; i < _np; i++) {
		std::vector<double> x(_n);
		switch (_mutation) {
		case tent_map: {
			for (int j = 0; j < _n; j++) {
				const double r = sampleLogistic();
				x[j] = _lower[j] + r * (_upper[j] - _lower[j]);
			}
			break;
		}
		default: {
			for (int j = 0; j < _n; j++) {
				x[j] = Random::get(_lower[j], _upper[j]);
			}
			break;
		}
		}
		const double fx = _f._f(&x[0]);
		const point part { x, fx };
		_pool.push_back(std::move(part));
		if (fx < _fgbest) {
			_fgbest = fx;
			std::copy(part._x.begin(), part._x.end(), _bestx.begin());
		}
	}
	_best = _fgbest;
	_fev = _np;

	// performance index for adapting number of sub-populations
	_nks = 0;
	for (int k = 1; k <= _np && _np >= _npmin * k; k++) {
		_nks++;
	}
	_perfindex = std::vector<double>(_nks, 0.);
	_pstrat = std::vector<double>(_nks, 1.);
}

void JayaSearch::iterate() {

	// divide population into K sub-populations
	divideSubpopulation();

	// apply the evolution algorithm to each sub-population
	_pbest = _best;
	_best = std::numeric_limits<double>::infinity();
	int idx_next_subpop = 0;
	for (int q = 0; q < _k; q++) {

		// get best and worst members
		auto *best = &_pool[idx_next_subpop];
		auto *worst = &_pool[idx_next_subpop];
		for (int i = idx_next_subpop; i < idx_next_subpop + _len[q]; i++) {
			if (_pool[i]._f < best->_f) {
				best = &_pool[i];
			}
			if (_pool[i]._f > worst->_f) {
				worst = &_pool[i];
			}
		}

		// perform update
		for (int i = idx_next_subpop; i < idx_next_subpop + _len[q]; i++) {
			evolve(i, *best, *worst);
		}
		idx_next_subpop += _len[q];
	}

	// update performance index
	if (_adapt_k) {
		const double improvement = (_pbest - _best)
				/ std::max(1e-12, std::fabs(_pbest));
		_perfindex[_k - 1] = improvement;
		_pstrat[_k - 1] = std::exp(_temper * _perfindex[_k - 1]);
		adaptK();
	}
}

multivariate_solution JayaSearch::solution(){
	return {_bestx, _fev, converged()};
}

multivariate_solution JayaSearch::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	bool converge = false;
	while (true) {
		iterate();

		// reached budget
		if (_fev >= _mfev) {
			break;
		}

		if (converged()){
			converge = true;
			break;
		}
	}
	return {_bestx, _fev, converge};
}

bool JayaSearch::converged(){

	// compute standard deviation of swarm radiuses
	int count = 0;
	double mean = 0.;
	double m2 = 0.;
	for (const auto &pt : _pool) {
		const double x = dnrm2(_n, &(pt._x)[0]);
		count++;
		const double delta = x - mean;
		mean += delta / count;
		const double delta2 = x - mean;
		m2 += delta * delta2;
	}

	// test convergence in standard deviation
	return m2 <= (_np - 1) * _stol * _stol;
}

/* =============================================================
 *
 * 				PARAMETER ADAPTATION SUBROUTINES
 *
 * =============================================================
 */
void JayaSearch::divideSubpopulation() {

	// allocate N_subpop = NP // K elements at random to each sub-population
	Random::shuffle(_pool.begin(), _pool.end());
	const int base_len = _np / _k;
	std::fill(_len.begin(), _len.end(), base_len);

	// fill the remaining sub-populations randomly
	int total_len = base_len * _k;
	while (total_len < _np) {
		const int idx = Random::get(0, _k - 1);
		_len[idx]++;
		total_len++;
	}
}

void JayaSearch::adaptK() {
	const double s = std::accumulate(_pstrat.begin(), _pstrat.end(), 0.);
	double U = Random::get(0., s);
	for (int k = 0; k < _nks; k++) {
		U -= _pstrat[k];
		if (U <= 0.) {
			_k = k + 1;
			return;
		}
	}
	_k = _nks;
}

/* =============================================================
 *
 * 					SWARM UPDATE SUBROUTINES
 *
 * =============================================================
 */
void JayaSearch::evolve(int i, point &best, point &worst) {

	// evolve the position
	auto &part = _pool[i];
	switch (_mutation) {
	case original: {
		for (int j = 0; j < _n; j++) {
			const double r1 = Random::get(0., 1.);
			const double r2 = Random::get(0., 1.);
			_tmp[j] = part._x[j] + r1 * (best._x[j] - std::fabs(part._x[j]))
					- r2 * (worst._x[j] - std::fabs(part._x[j]));
			_tmp[j] = std::max(_lower[j], std::min(_tmp[j], _upper[j]));
		}
		break;
	}
	case levy: {
		for (int j = 0; j < _n; j++) {
			const double step = sampleLevy();
			const double stepSize = _scale * step * (part._x[j] - best._x[j]);
			const double levy_ij = part._x[j] + stepSize * Random::get(0., 1.);
			const double r1 = Random::get(0., 1.);
			const double r2 = Random::get(0., 1.);
			_tmp[j] = levy_ij + r1 * (best._x[j] - std::fabs(part._x[j]))
					- r2 * (worst._x[j] - std::fabs(part._x[j]));
			_tmp[j] = std::max(_lower[j], std::min(_tmp[j], _upper[j]));
		}
		break;
	}
	case tent_map: {
		for (int j = 0; j < _n; j++) {
			double r1, r2;
			if (&_pool[i] == &best) {
				r1 = sampleTentMap();
				r2 = sampleTentMap();
			} else {
				r1 = Random::get(0., 1.);
				r2 = Random::get(0., 1.);
			}
			_tmp[j] = part._x[j] + r1 * (best._x[j] - std::fabs(part._x[j]))
					- r2 * (worst._x[j] - std::fabs(part._x[j]));
			_tmp[j] = std::max(_lower[j], std::min(_tmp[j], _upper[j]));
		}
		break;
	}
	case logistic: {
		for (int j = 0; j < _n; j++) {
			double r1, r2;
			if (&_pool[i] == &best) {
				r1 = sampleLogistic();
				r2 = sampleLogistic();
			} else {
				r1 = Random::get(0., 1.);
				r2 = Random::get(0., 1.);
			}
			_tmp[j] = part._x[j] + r1 * (best._x[j] - std::fabs(part._x[j]))
					- r2 * (worst._x[j] - std::fabs(part._x[j]));
			_tmp[j] = std::max(_lower[j], std::min(_tmp[j], _upper[j]));
		}
		break;
	}
	}

	// evaluate fitness of new position
	const double ftmp = _f._f(&_tmp[0]);
	_fev++;

	// copy the particle back to the swarm if it is an improvement
	if (ftmp < part._f) {
		std::copy(_tmp.begin(), _tmp.end(), part._x.begin());
		part._f = ftmp;
	}

	// update the global best for the current iteration
	_best = std::max(_best, part._f);
	if (part._f < _fgbest) {
		_fgbest = part._f;
		std::copy(part._x.begin(), part._x.end(), _bestx.begin());
	}
}

/* =============================================================
 *
 * 					SAMPLING SUBROUTINES
 *
 * =============================================================
 */
double JayaSearch::sampleLevy() {

	// Mantegna's algorithm
	const double sigma_v = 1.;
	const double u = Random::get(_Z) * _sigmau;
	const double v = Random::get(_Z) * sigma_v;
	return u / (std::pow(std::fabs(v), 1. / _beta));
}

double JayaSearch::sampleTentMap() {

	// sample tent map chaos
	if (_xchaos < 0.7) {
		_xchaos /= 0.7;
	} else {
		while (_xchaos == 0.7) {
			_xchaos = Random::get(0., 1.);
		}
		_xchaos = 10. / 3. * (1. - _xchaos);
	}
	return _xchaos;
}

double JayaSearch::sampleLogistic() {

	// sample logistic map chaos
	while (_xchaos == 0.5) {
		_xchaos = Random::get(0., 1.);
	}
	_xchaos = 4. * _xchaos * (1. - _xchaos);
	return _xchaos;
}
