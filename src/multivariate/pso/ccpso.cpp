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

 [1] Van den Bergh, Frans, and Andries Petrus Engelbrecht. "A cooperative approach
 to particle swarm optimization." IEEE transactions on evolutionary computation
 8.3 (2004): 225-239.

 [2] Li, Xiaodong, and Xin Yao. "Cooperatively coevolving particle swarms for
 large scale optimization." IEEE Transactions on Evolutionary Computation 16.2
 (2012): 210-224.

 [3] Li, Xiaodong, and Xin Yao. "Tackling high dimensional nonseparable optimization
 problems by cooperatively coevolving particle swarms." 2009 IEEE congress on
 evolutionary computation. IEEE, 2009.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <iostream>
#include <stdexcept>

#include "../../blas.h"
#include "../../random.hpp"

#include "ccpso.h"

using Random = effolkronium::random_static;

CCPSOSearch::CCPSOSearch(int mfev, double sigmatol, int np, int *pps, int npps,
		bool correct, double pcauchy, MultivariateOptimizer *local,
		int localfreq) {
	_stol = sigmatol;
	_np = np;
	_pps = std::vector<int>(pps, pps + npps);
	_npps = npps;
	_mfev = mfev;
	_correct = correct;
	_phat0 = pcauchy;
	_adaptp = !(_phat0 > 0. && _phat < 1.);
	_local = local;
	_localfreq = localfreq;
}

void CCPSOSearch::init(const multivariate_problem &f, const double *guess) {

	// initialize domain
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [CC-PSO]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// initialize algorithm primitives and initialize swarms' components
	_fev = 0;
	_X.clear();
	_X.resize(_np, std::vector<double>(_n));
	_Y.clear();
	_Y.resize(_np, std::vector<double>(_n));
	_yhat = std::vector<double>(_n);
	_fyhat = std::numeric_limits<double>::infinity();
	for (int ip = 0; ip < _np; ip++) {
		for (int i = 0; i < _n; i++) {
			_X[ip][i] = Random::get(_lower[i], _upper[i]);
			_Y[ip][i] = _X[ip][i];
		}
		const double fx = _f._f(&(_X[ip])[0]);
		if (fx < _fyhat) {
			_fyhat = fx;
			std::copy(_X[ip].begin(), _X[ip].end(), _yhat.begin());
		}
	}
	_fev = _np;
	_gen = 0;
	_temp = std::vector<double>(_n);
	_temp3 = std::vector<double>(3);
	_idx3 = std::vector<int>(3);
	_range = std::vector<int>(_n);
	_improved = false;
	_is = -1;
	if (_adaptp) {
		_phat = 0.5;
	} else {
		_phat = _phat0;
	}
}

void CCPSOSearch::iterate() {
	randomizeComponents();
	updateSwarm();
	updatePosition();
	if (_local && _localfreq > 0 && _gen % _localfreq == 0) {
		localSearch();
	}
	_gen++;
	std::cout << _fyhat << std::endl;
}

multivariate_solution CCPSOSearch::solution(){
	return {_yhat, _fev, converged()};
}

multivariate_solution CCPSOSearch::optimize(const multivariate_problem &f,
		const double *guess) {

	// initialization
	init(f, guess);

	// main loop
	bool converge = false;
	while (true) {
		iterate();

		// check max number of evaluations
		if (_fev >= _mfev) {
			break;
		}

		// test convergence in standard deviation
		if (converged()) {
			converge = true;
			break;
		}
	}
	return {_yhat, _fev, converge};
}

double CCPSOSearch::evaluate(int j, double *z) {

	// change the context yhat component j by z
	int k = 0;
	for (const int d : _k[j]) {
		_temp[d] = _yhat[d];
		_yhat[d] = z[k];
		k++;
	}

	// evaluate function at b
	const double fit = _f._f(&_yhat[0]);
	_fev++;

	// restore the context vector
	for (const int d : _k[j]) {
		_yhat[d] = _temp[d];
	}
	return fit;
}

bool CCPSOSearch::converged(){

	// estimate diversity of swarm
	int count = 0;
	double mean = 0.;
	double m2 = 0.;
	for (const auto &pt : _X) {
		const double x = dnrm2(_n, &pt[0]);
		count++;
		const double delta = x - mean;
		mean += delta / count;
		const double delta2 = x - mean;
		m2 += delta * delta2;
	}

	return m2 <= (_np - 1) * _stol * _stol;
}

/* =============================================================
 *
 * 				PARTICLE EVOLUTION SUBROUTINES
 *
 * =============================================================
 */
void CCPSOSearch::randomizeComponents() {

	// sample an s at random, the number of components per swarm
	const int is0 = _is;
	_is = sampleSubsetIndex();
	if (_is != is0) {
		_cpswarm = _pps[_is];
		if (_cpswarm <= 0 || _cpswarm > _n || _n % _cpswarm != 0) {
			throw std::invalid_argument(
					"Error [CC-PSO]: invalid component size.");
		}
		_nswarm = _n / _cpswarm;
		_k.clear();
		_k.resize(_nswarm, std::vector<int>(_cpswarm, 0));
		_fX.clear();
		_fX.resize(_nswarm, std::vector<double>(_np, 0.));
		_fY.clear();
		_fY.resize(_nswarm, std::vector<double>(_np, 0.));
		_z = std::vector<double>(_cpswarm);
		_ibest.clear();
		_ibest.resize(_nswarm, std::vector<int>(_np, 0));
		_strat.clear();
		_strat.resize(_nswarm, std::vector<int>(_np, 0));
		if (_local && _localfreq > 0) {
			_wlb = std::vector<double>(_nswarm);
			_wub = std::vector<double>(_nswarm);
			_wguess = std::vector<double>(_nswarm);
		}
	}

	// initialize the component indices for each swarm
	for (int j = 0; j < _n; j++) {
		_range[j] = j;
	}
	Random::shuffle(_range.begin(), _range.end());
	int i = 0;
	for (auto &k : _k) {
		for (int j = 0; j < _cpswarm; j++) {
			k[j] = _range[i];
			i++;
		}
	}
}

void CCPSOSearch::updateSwarm() {

	// re-evaluate particles
	for (int j = 0; j < _nswarm; j++) {
		const auto &coords = _k[j];
		for (int i = 0; i < _np; i++) {

			// re-evaluate current position
			for (int k = 0; k < _cpswarm; k++) {
				_z[k] = _X[i][coords[k]];
			}
			_fX[j][i] = evaluate(j, &_z[0]);

			// re-evaluate personal best
			for (int k = 0; k < _cpswarm; k++) {
				_z[k] = _Y[i][coords[k]];
			}
			_fY[j][i] = evaluate(j, &_z[0]);
		}
	}

	// update personal best position of all particles and global best of all swarms
	// and the global best position so far
	bool yhatupdate = false;
	std::copy(_yhat.begin(), _yhat.end(), _temp.begin());
	for (int j = 0; j < _nswarm; j++) {
		for (int i = 0; i < _np; i++) {

			// update the personal best P_j.y_i
			if (_fX[j][i] < _fY[j][i]) {
				for (const auto d : _k[j]) {
					_Y[i][d] = _X[i][d];
				}
			}

			// update the global best P_j.yhat
			if (_fY[j][i] < _fyhat) {
				for (const auto d : _k[j]) {
					_yhat[d] = _Y[i][d];
				}
				yhatupdate = true;
			}

			// find the local best P_j.yhat_i' in neighborhood topology
			_idx3[0] = (i - 1 + _np) % _np;
			_idx3[1] = i;
			_idx3[2] = (i + 1) % _np;
			for (int k = 0; k < 3; k++) {
				_temp3[k] = _fY[j][_idx3[k]];
			}
			const int imin = std::min_element(_temp3.begin(), _temp3.end())
					- _temp3.begin();
			_ibest[j][i] = _idx3[imin];
		}
	}

	// determine whether or not the global best point has improved
	const double fyhat0 = _fyhat;
	if (yhatupdate) {
		_fyhat = _f._f(&_yhat[0]);
		_fev++;
	}
	_improved = _fyhat < fyhat0;
	if (!_improved) {
		_fyhat = fyhat0;
		std::copy(_temp.begin(), _temp.end(), _yhat.begin());
	}

	// adapt success rate for evolution parameters
	if (_gen > 0 && _adaptp) {
		int cs = 0, ns = 0, ctot = 0, ntot = 0;
		for (int j = 0; j < _nswarm; j++) {
			for (int i = 0; i < _np; i++) {
				if (_fX[j][i] < _fY[j][i]) {
					if (_strat[j][i] == 0) {
						cs++;
					} else {
						ns++;
					}
				}
				if (_strat[j][i] == 0) {
					ctot++;
				} else {
					ntot++;
				}
			}
		}
		const double crate = 1. * cs / std::max(1, ctot);
		const double nrate = 1. * ns / std::max(1, ntot);
		_phat = std::max(0.05,
				std::min(crate / std::max(1., crate + nrate), 0.95));
	}
}

void CCPSOSearch::updatePosition() {
	for (int j = 0; j < _nswarm; j++) {
		for (int i = 0; i < _np; i++) {
			if (Random::get(0., 1.) < _phat) {

				// global exploration with Cauchy distribution
				for (auto d : _k[j]) {
					const double C1 = sampleCauchy();
					const int ihat = _ibest[j][i];
					const double sigma = std::fabs(_Y[i][d] - _Y[ihat][d]);
					_X[i][d] = _Y[i][d] + C1 * sigma;
					if (_correct) {
						_X[i][d] = std::max(_lower[d],
								std::min(_X[i][d], _upper[d]));
					}
				}
				_strat[j][i] = 0;
			} else {

				// local exploration with Normal distribution
				for (auto d : _k[j]) {
					const double N01 = Random::get(_Z);
					const int ihat = _ibest[j][i];
					const double sigma = std::fabs(_Y[i][d] - _Y[ihat][d]);
					_X[i][d] = _Y[ihat][d] + N01 * sigma;
					if (_correct) {
						_X[i][d] = std::max(_lower[d],
								std::min(_X[i][d], _upper[d]));
					}
				}
				_strat[j][i] = 1;
			}
		}
	}
}

void CCPSOSearch::localSearch() {

	// compute new bounding box
	const double inf = std::numeric_limits<double>::infinity();
	std::fill(_wlb.begin(), _wlb.end(), -inf);
	std::fill(_wub.begin(), _wub.end(), +inf);
	for (int j = 0; j < _nswarm; j++) {
		for (const auto k : _k[j]) {
			double scale = _yhat[k];
			if (std::fabs(scale) < 1e-3) {
				if (scale > 0.) {
					scale = 1e-3;
				} else {
					scale = -1e-3;
				}
			}
			double lbk = _lower[k] / scale;
			double ubk = _upper[k] / scale;
			if (lbk > ubk) {
				const double temp = lbk;
				lbk = ubk;
				ubk = temp;
			}
			_wlb[j] = std::max(_wlb[j], lbk);
			_wub[j] = std::min(_wub[j], ubk);
		}
		_wguess[j] = std::max(_wlb[j], std::min(1., _wub[j]));
	}

	// local optimization problem
	const multivariate &faux = [&](const double *w) -> double {
		for (int j = 0; j < _nswarm; j++) {
			for (const auto k : _k[j]) {
				_temp[k] = _yhat[k] * w[j];
			}
		}
		return _f._f(&_temp[0]);
	};

	// solution
	const multivariate_problem paux { faux, _nswarm, &_wlb[0], &_wub[0] };
	const auto &sol = _local->optimize(paux, &_wguess[0]);
	const auto &w = sol._sol;
	_fev += sol._fev;

	// reject a solution that does not lie in the bounds
	if (_correct) {
		for (int j = 0; j < _nswarm; j++) {
			for (const auto k : _k[j]) {
				if (_yhat[k] * w[j] < _lower[k]
						|| _yhat[k] * w[j] > _upper[k]) {
					return;
				}
			}
		}
	}

	// replace yhat with the local solution
	const double fwy = faux(&w[0]);
	_fev++;
	if (fwy < _fyhat) {
		std::copy(_temp.begin(), _temp.end(), _yhat.begin());
		_fyhat = fwy;
		_improved = true;
	}
}

/* =============================================================
 *
 * 					SAMPLING SUBROUTINES
 *
 * =============================================================
 */
double CCPSOSearch::sampleCauchy() {
	return std::tan(M_PI * (Random::get(0., 1.) - 0.5));
}

int CCPSOSearch::sampleSubsetIndex() {
	if (_improved) {
		return _is;
	} else {
		return Random::get(0, _npps - 1);
	}
}
