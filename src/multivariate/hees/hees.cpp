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

 [1] Glasmachers, Tobias, and Oswin Krause. "The Hessian Estimation Evolution Strategy."
 International Conference on Parallel Problem Solving from Nature. Springer, Cham, 2020.
 */

#include <cmath>
#include <iostream>

#include "../../blas.h"
#include "../../random.hpp"

#include "hees.h"

using Random = effolkronium::random_static;

Hees::Hees(int mfev, double tol, int mres, bool print, int np, double sigma0) {
	_mfev = mfev;
	_tol = tol;
	_mres = mres;
	_print = print;
	_mu = np;
	_sigma0 = sigma0;
	_adaptpop = np <= 0;
}

void Hees::init(const multivariate_problem &f, const double *guess) {

	// initialize domain
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [HEES]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// first point
	_xmean = std::vector<double>(guess, guess + _n);
	_fm = _f._f(&_xmean[0]);
	_fev = 1;
	_xbest = std::vector<double>(_xmean);
	_fbest = _fm;

	// adaptive parameters
	if (_adaptpop) {
		_mu = static_cast<int>(2. + 1.5 * std::log(1. * _n));
	}
	_B = static_cast<int>(std::ceil((1. * _mu) / _n));
	_np = _B * _n;
	_sigma = _sigma0;
	_kappa = 3.;
	_etaA = 0.5;
	_gs = 0.;
	_chi = std::sqrt(1. * _n) * (1. - 1. / (4. * _n) + 1. / (21. * _n * _n));

	// weights
	double wsum = 0.;
	_weights = std::vector<double>(2 * _mu);
	for (int i = 0; i < _mu * 2; i++) {
		_weights[i] = std::log(_mu + 0.5)
				- std::log(std::min(1. + i, _mu + 0.5));
		wsum += _weights[i];
	}
	dscalm(_mu * 2, 1. / wsum, &_weights[0], 1);

	// compute other parameters
	_mueff = 1.
			/ std::inner_product(_weights.begin(), _weights.end(),
					_weights.begin(), 0.);
	_mueffm = 1. / (1. / _mueff - 1. / (2. * _mu - 1.) * (1. - 1. / _mueff));
	_cs = (_mueffm + 2.) / (_n + _mueffm + 3.);
	_ds = 1. + _cs
			+ 2. * std::max(0., std::sqrt((_mueff - 1.) / (_n + 1.)) - 1.);

	// other memory
	_ps = std::vector<double>(_n, 0.);
	_norms = std::vector<double>(_np);
	_hess = std::vector<double>(_mu);
	_q = std::vector<double>(_mu);
	_dz = std::vector<double>(_n);
	_a.clear();
	_a.resize(_n, std::vector<double>(_n, 0.));
	for (int i = 0; i < _n; i++) {
		_a[i][i] = 1.;
	}
	_aold.clear();
	_aold.resize(_n, std::vector<double>(_n, 0.));
	_b.clear();
	_b.resize(_np, std::vector<double>(_n, 0.));
	_x.clear();
	_x.resize(_mu * 2, std::vector<double>(_n, 0.));
	_g.resize(_n, std::vector<double>(_n, 0.));
	_fitness.clear();
	for (int i = 0; i < _mu * 2; i++) {
		const hees_index index { 0, 0, 0. };
		_fitness.push_back(std::move(index));
	}
}

void Hees::iterate() {
	samplePopulation();
	evaluateAndSortPopulation();
	covarianceUpdate();
	meanUpdate();
	stepSizeUpdate();
}

multivariate_solution Hees::solution(){
	return {_xbest, _fev, converged()};
}

multivariate_solution Hees::optimize(const multivariate_problem &f,
		const double *guess) {

	// single run
	if (_mres <= 1) {
		init(f, guess);
		bool conv = false;
		while (_fev < _mfev) {
			iterate();
			if (converged()) {
				conv = true;
				break;
			}
		}
		return {_xbest, _fev, conv};
	}

	// print header
	_table = Tabular();
	if (_print) {
		_table.setWidth( { 5, 25, 10 });
		_table.printRow("iter", "f*", "fev");
	}

	// multiple runs
	if (_adaptpop) {
		_mu = static_cast<int>(2. + 1.5 * std::log(1. * f._n));
	}
	_fev = 0;
	double fbest = std::numeric_limits<double>::infinity();
	std::vector<double> x0(guess, guess + f._n);
	for (int res = 1; res <= _mres; res++) {

		// run the local solver
		Hees local { _mfev - _fev, _tol, 1, false, _mu, _sigma0 };
		const auto &sol = local.optimize(f, &x0[0]);

		// update best point so far
		if (local._fbest < fbest) {
			fbest = local._fbest;
			_xbest = sol._sol;
		}

		// update budget
		_fev += sol._fev;

		// print progress
		if (_print) {
			_table.printRow(res, fbest, _fev);
		}
		if (_fev >= _mfev) {
			break;
		}

		// double population size
		_mu <<= 1;

		// randomize next point
		for (int i = 0; i < f._n; i++) {
			x0[i] = Random::get(f._lower[i], f._upper[i]);
		}
	}
	return {_xbest, _fev, false};
}

void Hees::samplePopulation() {

	// sample N(0, 1) points
	for (int i = 0; i < _np; i++) {
		for (int j = 0; j < _n; j++) {
			_b[i][j] = Random::get(_Z);
		}
		_norms[i] = dnrm2(_n, &(_b[i])[0]);
	}

	// Gram-Schmidt procedure to orthonormalize samples
	for (int j = 0; j < _B; j++) {
		for (int i = 0; i < _n; i++) {
			auto &vij = _b[_n * j + i];
			for (int k = 0; k < i; k++) {
				auto &vkj = _b[_n * j + k];
				const double dt = std::inner_product(vkj.begin(), vkj.end(),
						vij.begin(), 0.);
				daxpym(_n, -dt, &vkj[0], 1, &vij[0], 1);
			}
			const double nij = dnrm2(_n, &vij[0]);
			dscalm(_n, 1. / nij, &vij[0], 1);
		}
	}

	// rescale vectors by their norm
	for (int i = 0; i < _np; i++) {
		dscalm(_n, _norms[i], &(_b[i])[0], 1);
	}

	// compute points using mirrored sampling
	for (int p = 0; p < _mu; p++) {
		for (int i = 0; i < _n; i++) {
			const double dot = std::inner_product(_a[i].begin(), _a[i].end(),
					_b[p].begin(), 0.);
			_x[p][i] = _xmean[i] - _sigma * dot;
			_x[p + _mu][i] = _xmean[i] + _sigma * dot;
		}
	}
}

void Hees::evaluateAndSortPopulation() {

	// evaluate function values
	for (int i = 0; i < _mu * 2; i++) {
		_fitness[i]._index = i;
		_fitness[i]._value = _f._f(&(_x[i])[0]);
	}
	_fev += _mu * 2;

	// rank population by fitness
	std::sort(_fitness.begin(), _fitness.end(), hees_index::compare_fitness);
	for (int i = 0; i < _mu * 2; i++) {
		_fitness[i]._rank = i;
	}

	// reset the order
	// two sorts have time O(n*log(n)) as opposed to O(n^2) for naive rank
	std::sort(_fitness.begin(), _fitness.end(), hees_index::compare_index);
}

void Hees::covarianceUpdate() {

	// estimate hessian
	double maxh = -std::numeric_limits<double>::infinity();
	for (int i = 0; i < _mu; i++) {
		_hess[i] = (_fitness[i + _mu]._value + _fitness[i]._value - 2. * _fm)
				/ (_norms[i] * _norms[i]);
		maxh = std::max(maxh, _hess[i]);
	}
	if (maxh <= 0.) {
		return;
	}

	// trust region size
	const double ctrust = maxh / _kappa;

	// truncate to the trust region
	double meanq = 0.;
	for (int i = 0; i < _mu; i++) {
		_hess[i] = std::max(_hess[i], ctrust);
		_q[i] = std::log(_hess[i]);
		meanq += _q[i] / _mu;
	}

	// subtract mean to ensure unit determinant
	// apply the learning rate and inverse square root adjustment
	for (int i = 0; i < _mu; i++) {
		_q[i] -= meanq;
		_q[i] *= (-_etaA * 0.5);
		_q[i] = std::exp(_q[i]);
	}

	// compute the multiplicative update matrix
	for (int r = 0; r < _n; r++) {
		for (int c = 0; c < _n; c++) {
			_g[r][c] = 0.;
			for (int i = 0; i < _np; i++) {
				if (i < _mu) {
					_g[r][c] += _q[i] / (_norms[i] * _norms[i] * _B) * _b[i][r]
							* _b[i][c];
				} else {
					_g[r][c] += 1. / (_norms[i] * _norms[i] * _B) * _b[i][r]
							* _b[i][c];
				}
			}
		}
	}

	// matrix adaptation
	for (int i = 0; i < _n; i++) {
		std::copy(_a[i].begin(), _a[i].end(), _aold[i].begin());
	}
	for (int r = 0; r < _n; r++) {
		for (int c = 0; c < _n; c++) {
			_a[r][c] = 0.;
			for (int l = 0; l < _n; l++) {
				_a[r][c] += _aold[r][l] * _g[l][c];
			}
		}
	}
}

void Hees::meanUpdate() {

	// update the mean vector
	std::fill(_xmean.begin(), _xmean.end(), 0.);
	for (int i = 0; i < _mu * 2; i++) {
		const int k = _fitness[i]._rank;
		daxpym(_n, _weights[k], &(_x[i])[0], 1, &_xmean[0], 1);
	}
	_fm = _f._f(&_xmean[0]);
	_fev++;

	// update best point
	if (_fm < _fbest) {
		_fbest = _fm;
		std::copy(_xmean.begin(), _xmean.end(), _xbest.begin());
	}
}

void Hees::stepSizeUpdate() {

	// update p_s
	std::fill(_dz.begin(), _dz.end(), 0.);
	for (int i = 0; i < _mu; i++) {
		const int rankm = _fitness[i]._rank;
		const int rankp = _fitness[i + _mu]._rank;
		const double wrankm = _weights[rankm];
		const double wrankp = _weights[rankp];
		daxpym(_n, -wrankm, &(_b[i])[0], 1, &_dz[0], 1);
		daxpym(_n, +wrankp, &(_b[i])[0], 1, &_dz[0], 1);
	}
	const double csc = std::sqrt(_cs * (2. - _cs) * _mueffm);
	for (int i = 0; i < _n; i++) {
		_ps[i] = (1. - _cs) * _ps[i] + csc * _dz[i];
	}

	// update step size sigma
	_gs = std::pow(1. - _cs, 2.) * _gs + _cs * (2. - _cs);
	const double psn = dnrm2(_n, &_ps[0]);
	const double s = psn / _chi - std::sqrt(_gs);
	_sigma *= std::exp(std::min(1., _cs / _ds * s));
}

bool Hees::converged() {

	// compute standard deviation of swarm radiuses
	int count = 0;
	double mean = 0.;
	double m2 = 0.;
	for (const auto &pt : _fitness) {
		count++;
		const double delta = pt._value - mean;
		mean += delta / count;
		const double delta2 = pt._value - mean;
		m2 += delta * delta2;
	}

	// test convergence in standard deviation
	return m2 <= count * _tol * _tol;
}
