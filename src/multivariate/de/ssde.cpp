/*
 Copyright (c) 2024 Mike Gimelfarb

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

 [1] Kumar, A., Misra, R. K., Singh, D., Mishra, S., & Das, S. (2019).
 The spherical search algorithm for bound-constrained global optimization
 problems. Applied Soft Computing, 85, 105734.

 [2] Zhao, J., Zhang, B., Guo, X., Qi, L., & Li, Z. (2022). Self-adapting
 spherical search algorithm with differential evolution for global optimization.
 Mathematics, 10(23), 4519.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include "../../blas.h"
#include "../../math_utils.h"
#include "../../random.hpp"

#include "ssde.h"

using Random = effolkronium::random_static;

SSDESearch::SSDESearch(int mfev, int npinit, double tol, int patience,
		int npmin, double ptop, int h, bool usede, bool repaircr) {
	_mfev = mfev;
	_npinit = npinit;
	_tol = tol;
	_patience = patience;
	_ptop = ptop;
	_npmin = npmin;
	_H = h;
	_usede = usede;
	_repaircr = repaircr;
}

void SSDESearch::init(const multivariate_problem &f, const double *guess) {

	// define problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [SSDE]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// define parameters and memory
	_fev = 0;
	_convcount = 0;
	_A.clear();
	for (int i = 0; i < _n; i++) {
		_A.push_back(std::vector<double>(_n, 0.));
	}
	_b = std::vector<double>(_n, 0.);
	_z = std::vector<double>(_n, 0.);
	_y = std::vector<double>(_n, 0.);
	_work = std::vector<double>(_n);
	_iwork = std::vector<int>(_n);
	for (int i = 0; i < _n; i++) {
		_iwork[i] = i;
	}

	// define swarm within search bounds
	_np = _npinit;
	_swarm.clear();
	for (int i = 0; i < _np; i++) {
		std::vector<double> x(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = Random::get(_lower[j], _upper[j]);
		}
		const point part { x, _f._f(&x[0]) };
		_swarm.push_back(std::move(part));
		++_fev;
	}

	// define additional points using opposition based learning
	if (_usede) {
		for (int i = 0; i < _np; i++) {
			std::vector<double> x(_n);
			for (int j = 0; j < _n; j++) {
				x[j] = _lower[j] + _upper[j] - _swarm[i]._x[j];
			}
			const point part { x, _f._f(&x[0]) };
			_swarm.push_back(std::move(part));
			++_fev;
		}
		_np += _npinit;
	}

	// rank population by fitness and keep top np points
	std::sort(_swarm.begin(), _swarm.end(), point::compare_fitness);
	while (_np > _npinit) {
		_swarm.pop_back();
		--_np;
	}

	// for parameter adaptation
	_L1 = std::vector<double>(_H, 0.5);
	_L2 = std::vector<double>(_H, 0.5);
	_LCR = std::vector<double>(_H, 0.5);
	_SR.clear();
	_SC.clear();
	_SCR.clear();
	_w.clear();
	_wCR.clear();
	_k = 1;
	_kCR = 1;
}

void SSDESearch::iterate() {
	const double oldbestf = _swarm[0]._f;

	// compute the orthogonal matrix A
	randA();

	// clear adaptation memory
	_SR.clear();
	_SC.clear();
	_SCR.clear();
	_w.clear();
	_wCR.clear();

	// main update loop
	for (int i = 0; i < _np; i++) {

		// sample element from parameter memory
		const int iL = Random::get(0, _H - 1);

		// compute the random binary vector
		double prank = Random::get(_Z) * 0.1 + _L1[iL];
		prank = std::max(0., std::min(1., prank));
		for (int r = 0; r < _n; r++) {
			if (Random::get(0., 1.) < prank) {
				_b[r] = 1.;
			} else {
				_b[r] = 0.;
			}
		}

		// sample 3 distinct members
		int pi, qi, ri;
		sample3(i, pi, qi, ri);

		// top p percent candidate
		const int itop = std::max(1, static_cast<int>(_ptop * _np));
		const int pbest = Random::get(0, itop);

		// R parameter
		const double R = (1. * _fev) / _mfev;

		// strategy selection
		if (_usede) {

			// DE-spherical search
			if (R < 0.333) {

				// explore: generate trial point towards the random points
				for (int d = 0; d < _n; d++) {
					_z[d] = _swarm[pi]._x[d] + _swarm[qi]._x[d]
							- _swarm[ri]._x[d] - _swarm[i]._x[d]
							+ R * (_swarm[pbest]._x[d] - _swarm[qi]._x[d]);
				}
			} else if (R < 0.666) {

				// balance: generate trial point towards the top p percent best
				for (int d = 0; d < _n; d++) {
					_z[d] = _swarm[pbest]._x[d] + _swarm[qi]._x[d]
							- _swarm[ri]._x[d] - _swarm[i]._x[d]
							+ R * (_swarm[pbest]._x[d] - _swarm[qi]._x[d]);
				}
			} else {

				// exploit: generate trial point towards the best
				for (int d = 0; d < _n; d++) {
					_z[d] = _swarm[0]._x[d] + _swarm[qi]._x[d]
							- _swarm[ri]._x[d] - _swarm[i]._x[d]
							+ R * (_swarm[pbest]._x[d] - _swarm[qi]._x[d]);
				}
			}
		} else {

			// vanilla spherical search
			if (i < 0.5 * _np) {

				// generate trial point towards the random points
				for (int d = 0; d < _n; d++) {
					_z[d] = _swarm[pi]._x[d] + _swarm[qi]._x[d]
							- _swarm[ri]._x[d] - _swarm[i]._x[d];
				}
			} else {

				// generate trial point towards the random top p candidate
				for (int d = 0; d < _n; d++) {
					_z[d] = _swarm[pbest]._x[d] + _swarm[qi]._x[d]
							- _swarm[ri]._x[d] - _swarm[i]._x[d];
				}
			}
		}

		// sample a step size control factor
		double ci = sampleCauchy() * 0.1 + _L2[iL];
		while (ci <= 0.) {
			ci = sampleCauchy() * 0.1 + _L2[iL];
		}
		ci = std::min(1., ci);

		// compute the trial point y
		computeTrialPoint(i, ci);
		const double fy = _f._f(&_y[0]);
		++_fev;

		// greedy selection
		if (fy <= _swarm[i]._f) {

			// memory management
			if (fy < _swarm[i]._f) {
				_SR.push_back(prank);
				_SC.push_back(ci);
				_w.push_back(_swarm[i]._f - fy);
			}

			// replacement
			std::copy(_y.begin(), _y.end(), _swarm[i]._x.begin());
			_swarm[i]._f = fy;

		} else if (_usede) {

			// sample a CR parameter
			const double CRi = std::max(0.,
					std::min(1., Random::get(_Z) * 0.1 + _LCR[iL]));

			// generate new candidate using DE
			double cr1 = 0.;
			const int j0 = Random::get(0, _n - 1);
			for (int d = 0; d < _n; d++) {
				if (d == j0 || Random::get(0., 1.) <= CRi) {
					_work[d] = _swarm[pi]._x[d]
							+ R * (_swarm[0]._x[d] - _swarm[qi]._x[d])
							+ R * (_swarm[0]._x[d] - _swarm[ri]._x[d]);
					if (_work[d] < _lower[d] || _work[d] > _upper[d]) {
						_work[d] = Random::get(_lower[d], _upper[d]);
					}
					cr1 += 1.;
				} else {
					_work[d] = _swarm[i]._x[d];
				}
			}
			if (_repaircr) {
				cr1 = cr1 / _n;
			} else {
				cr1 = CRi;
			}

			// greedy selection
			const double fu = _f._f(&_work[0]);
			++_fev;
			if (fu <= _swarm[i]._f) {

				// memory management
				if (fu < _swarm[i]._f) {
					_SCR.push_back(cr1);
					_wCR.push_back(_swarm[i]._f - fu);
				}

				// replacement
				std::copy(_work.begin(), _work.end(), _swarm[i]._x.begin());
				_swarm[i]._f = fu;
			}
		}
	}

	// memory update
	const int lenS = static_cast<int>(_SR.size());
	if (lenS > 0) {

		// compute mean rank and mean C
		double meanRnum = 0.0;
		double meanRden = 0.0;
		double meanCnum = 0.0;
		double meanCden = 0.0;
		for (int i = 0; i < lenS; i++) {
			meanRnum += _w[i] * _SR[i] * _SR[i];
			meanRden += _w[i] * _SR[i];
			meanCnum += _w[i] * _SC[i] * _SC[i];
			meanCden += _w[i] * _SC[i];
		}
		const double meanR = meanRnum / meanRden;
		const double meanC = meanCnum / meanCden;

		// replace in memory
		_L1[_k - 1] = meanR;
		_L2[_k - 1] = meanC;
		++_k;
		if (_k > _H) {
			_k = 1;
		}
	}

	// memory update for CR
	const int lenSCR = static_cast<int>(_SCR.size());
	if (_usede && lenSCR > 0) {

		// compute mean rank and mean C
		double meanCRnum = 0.0;
		double meanCRden = 0.0;
		for (int i = 0; i < lenSCR; i++) {
			meanCRnum += _wCR[i] * _SCR[i] * _SCR[i];
			meanCRden += _wCR[i] * _SCR[i];
		}
		const double meanCR = meanCRnum / meanCRden;

		// replace in memory
		_LCR[_kCR - 1] = meanCR;
		++_kCR;
		if (_kCR > _H) {
			_kCR = 1;
		}
	}

	// rank population by fitness
	std::sort(_swarm.begin(), _swarm.end(), point::compare_fitness);

	// increment number of iterations last improvement
	if (_swarm[0]._f < oldbestf) {
		_convcount = 0;
	} else {
		++_convcount;
	}

	// exponential population size reduction
	const int npnew = static_cast<int>(_npinit
			+ (_npmin - _npinit) * (1. * _fev) / _mfev);
	while (_np > npnew) {
		_swarm.pop_back();
		--_np;
	}
}

multivariate_solution SSDESearch::solution() {
	std::sort(_swarm.begin(), _swarm.end(), point::compare_fitness);
	return {_swarm[0]._x, _fev, converged()};
}

multivariate_solution SSDESearch::optimize(const multivariate_problem &f,
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
	std::sort(_swarm.begin(), _swarm.end(), point::compare_fitness);
	return {_swarm[0]._x, _fev, converge};
}

bool SSDESearch::converged() {

	// compute standard deviation of swarm radiuses
	int count = 0;
	double mean = 0.;
	double m2 = 0.;
	for (const auto &pt : _swarm) {
		const double x = dnrm2(_n, &pt._x[0]);
		count++;
		const double delta = x - mean;
		mean += delta / count;
		const double delta2 = x - mean;
		m2 += delta * delta2;
	}

	// test convergence in standard deviation
	if (m2 <= (_np - 1) * _tol * _tol) {
		return true;
	}

	// test convergence in progress
	if (_convcount > _patience) {
		return true;
	}
	return false;
}

void SSDESearch::randA() {

	// initialize A to the identity matrix
	Random::shuffle(_iwork.begin(), _iwork.end());
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j <= i; j++) {
			if (i == j) {
				_A[i][j] = 1.;
			} else {
				_A[i][j] = _A[j][i] = 0.;
			}
		}
	}

	// apply an orthogonal transform to A
	for (int ii = 1; ii <= _n / 2; ii++) {
		const int i = 2 * (ii - 1);
		_A[_iwork[i]][_iwork[i]] = std::sin(1e-12);
		_A[_iwork[i + 1]][_iwork[i + 1]] = _A[_iwork[i]][_iwork[i]];
		_A[_iwork[i]][_iwork[i + 1]] = std::cos(1e-12);
		_A[_iwork[i + 1]][_iwork[i]] = -_A[_iwork[i]][_iwork[i + 1]];
	}
}

void SSDESearch::computeTrialPoint(const int i, const double ci) {

	// compute work = diag(b) * A * z
	for (int d = 0; d < _n; d++) {
		_work[d] = 0.;
		for (int k = 0; k < _n; k++) {
			_work[d] += _A[d][k] * _z[k];
		}
		_work[d] *= _b[d];
	}

	// compute x[i] + c * A' * diag(b) * A * z
	for (int d = 0; d < _n; d++) {
		_y[d] = _swarm[i]._x[d];
		for (int k = 0; k < _n; k++) {
			_y[d] += ci * _A[k][d] * _work[k];
		}
		_y[d] = std::max(_lower[d], std::min(_upper[d], _y[d]));
	}
}

double SSDESearch::sampleCauchy() {
	return std::tan(M_PI * (Random::get(0., 1.) - 0.5));
}

void SSDESearch::sample3(const int i, int &pi, int &qi, int &ri) {
	pi = i;
	while (pi == i) {
		pi = Random::get(0, _np - 1);
	}
	qi = i;
	while (qi == i || qi == pi) {
		qi = Random::get(0, _np - 1);
	}
	ri = i;
	while (ri == i || ri == pi || ri == qi) {
		ri = Random::get(0, _np - 1);
	}
}
