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
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include "../../blas.h"

#include "base_cmaes.h"

void cmaes_history::add(double value) {
	_buffer = (_buffer + 1) % _cap;
	_hist[_buffer] = value;
	if (_len < _cap) {
		_len++;
	}
}

double cmaes_history::get(int i) {
	const int idx = (_cap + _buffer - i) % _cap;
	return _hist[idx];
}

BaseCmaes::BaseCmaes(int mfev, double tol, int np, double sigma0, bool bound) {
	_tol = tol;
	_lambda = np;
	_sigma0 = sigma0;
	_mfev = mfev;
	_adaptpop = _adaptit = false;
	_bound = bound;
}

void BaseCmaes::init(const multivariate_problem &f, const double *guess) {

	// initialize domain
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [CMA]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// adaptive initialization of population size
	if (_adaptpop) {
		_lambda = 4 + static_cast<int>(3. * std::log(_n));
	}
	_mu = _lambda / 2;

	// adaptive initialization of maximum evaluations
	if (_adaptit) {
		_mit = static_cast<int>(100
				+ 50 * (_n + 3) * (_n + 3) / std::sqrt(1. * _lambda));
		_mfev = _mit * _lambda;
	} else {
		_mit = _mfev / _lambda;
	}

	// initialization of population and ranking memory
	_arx.clear();
	_arx.resize(_lambda, std::vector<double>(_n, 0.));
	_ibw = std::vector<int>(4);
	std::fill(_ibw.begin(), _ibw.end(), 0);
	_ybw = std::vector<double>(4, 0.);
	_fitness.clear();
	for (int i = 0; i < _lambda; i++) {
		const cmaes_index index { 0, 0. };
		_fitness.push_back(std::move(index));
	}

	// initialize array for weighted recombination
	_weights = std::vector<double>(_mu);
	double sum = 0.;
	for (int i = 0; i < _mu; i++) {
		_weights[i] = std::log(0.5 * (_lambda + 1.)) - std::log(i + 1.);
		sum += _weights[i];
	}
	dscalm(_mu, 1. / sum, &_weights[0], 1);

	// initialize variance-effectiveness of sum w_i x_i
	const double lenw = std::inner_product(_weights.begin(), _weights.end(),
			_weights.begin(), 0.);
	_mueff = 1. / lenw;

	// initialize strategy parameter settings
	_chi = std::sqrt(_n) * (1. - 1. / (4. * _n) + 1. / (21. * _n * _n));
	_sigma = _sigma0;
	_cc = (4. + _mueff / _n) / (_n + 4. + 2. * _mueff / _n);
	_cs = (_mueff + 2.) / (5. + _n + _mueff);
	_c1 = 2. / ((1.3 + _n) * (1.3 + _n) + _mueff);
	_cmu = std::min(1. - _c1,
			2. * (_mueff - 2. + 1. / _mueff)
					/ ((2. + _n) * (2. + _n) + _mueff));
	_damps = 1. + _cs
			+ 2. * std::max(0., std::sqrt((_mueff - 1.) / (_n + 1.)) - 1.);

	// initialize other memories
	_pc = std::vector<double>(_n, 0.);
	_ps = std::vector<double>(_n, 0.);
	_artmp = std::vector<double>(_n, 0.);
	_xold = std::vector<double>(_n, 0.);
	_xmean = std::vector<double>(guess, guess + _n);
	_it = _fev = 0;

	// initialize history and convergence parameters
	_hlen = 10 + static_cast<int>(std::ceil((30. * _n) / _lambda));
	_ik = static_cast<int>(std::ceil(0.1 + _lambda / 4.));
	_best = cmaes_history { _hlen, -1, 0, std::vector<double>(_hlen, 0.) };
	_kth = cmaes_history { _hlen, -1, 0, std::vector<double>(_hlen, 0.) };
	_fbest = -std::numeric_limits<double>::infinity();
	_fworst = std::numeric_limits<double>::infinity();
}

void BaseCmaes::setParams(int np, double sigma, int mfev) {

	// used for IPOP and BIPOP strategies
	_lambda = np;
	_sigma0 = sigma;
	_mfev = mfev;
	_adaptpop = _adaptit = false;
	if (_bound) {
		_bound = false;
		std::cerr << "Warning [CMA]: box bounding is no longer enabled."
				<< std::endl;
	}
}

void BaseCmaes::iterate() {
	samplePopulation();
	evaluateAndSortPopulation();
	updateDistribution();
	updateHistory();
	_it++;
}

multivariate_solution BaseCmaes::solution(){
	return {bestSolution(), _fev, converged()};
}

multivariate_solution BaseCmaes::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	bool converge = false;
	while (_fev < _mfev) {
		iterate();
		if (converged()) {
			converge = true;
			break;
		}
	}
	return {bestSolution(), _fev, converge};
}

void BaseCmaes::updateSigma() {

	// basic sigma update
	const double pslen = dnrm2(_n, &_ps[0]);
	_sigma *= std::exp(std::min(1., (_cs / _damps) * (pslen / _chi - 1.)));

	// Adjust step size in case of equal function values (flat fitness)
	if (_fitness[0]._value == _fitness[_ik]._value) {
		_sigma *= std::exp(0.2 + _cs / _damps);
	}
	if (_it >= _hlen && _fworst - _fbest == 0.) {
		_sigma *= std::exp(0.2 + _cs / _damps);
	}
}

void BaseCmaes::updateHistory() {
	if (_it >= _mit) {
		return;
	}

	// append new observation
	_best.add(_fitness[0]._value);
	_kth.add(_fitness[_ik]._value);

	// update running recent worst and best fitness values
	if (_best._len == _best._cap) {
		_fbest = std::numeric_limits<double>::infinity();
		_fworst = -std::numeric_limits<double>::infinity();
		for (double fx : _best._hist) {
			_fbest = std::min(fx, _fbest);
			_fworst = std::max(fx, _fworst);
		}
	}
}

void BaseCmaes::evaluateAndSortPopulation() {

	// Sort by fitness
	for (int i = 0; i < _lambda; i++) {
		_fitness[i]._index = i;
		_fitness[i]._value = _f._f(&(_arx[i])[0]);
	}
	_fev += _lambda;

	// get the best and worst elements
	std::sort(_fitness.begin(), _fitness.end(), cmaes_index::compare_fitness);
	_ibw[0] = _fitness[0]._index;
	_ibw[1] = _fitness[1]._index;
	_ibw[2] = _fitness[_lambda - 2]._index;
	_ibw[3] = _fitness[_lambda - 1]._index;
	_ybw[0] = _fitness[0]._value;
	_ybw[1] = _fitness[1]._value;
	_ybw[2] = _fitness[_lambda - 2]._value;
	_ybw[3] = _fitness[_lambda - 1]._value;
}

std::vector<double> BaseCmaes::bestSolution() {
	if (_it <= 0) {
		return _xmean;
	} else {
		return _arx[_ibw[0]];
	}
}

