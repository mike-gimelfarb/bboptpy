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

 [1] Zhan, Zhi-Hui, Jun Zhang, Yun Li, and Henry Shu-Hung Chung. "Adaptive
 particle swarm optimization." IEEE Transactions on Systems, Man, and Cybernetics,
 Part B (Cybernetics) 39, no. 6 (2009): 1362-1381.
 */

#include <numeric>
#include <stdexcept>
#include <iostream>

#include "../../blas.h"
#include "../../random.hpp"

#include "apso.h"

using Random = effolkronium::random_static;

APSOSearch::APSOSearch(int mfev, double tol, int np, bool correct) {
	_tol = tol;
	_np = np;
	_mfev = mfev;
	_correct = correct;
}

void APSOSearch::init(const multivariate_problem &f, const double *guess) {

	// set problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [PSO]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// initialize parameters
	_w = 0.9;
	_smin = 0.1;
	_smax = 1.0;
	_vdamp = 0.2;
	_c1 = 2.;
	_c2 = 2.;
	_state = 0;
	_maxit = static_cast<int>(std::round(_mfev / (1. + _np)));
	_it = 0;

	// initialize swarm
	_fev = 0;
	_swarm.clear();
	_fbest = std::numeric_limits<double>::infinity();
	_xbest = std::vector<double>(_n);
	for (int i = 0; i < _np; i++) {

		// create particle
		std::vector<double> x(_n);
		std::vector<double> v(_n);
		std::vector<double> xb(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = Random::get(_lower[j], _upper[j]);
			xb[j] = x[j];
			v[j] = 0.;
		}
		const double fx = _f._f(&x[0]);
		_fev++;
		const apso_particle part { x, v, xb, fx, fx };
		_swarm.push_back(std::move(part));

		// update best and worst positions
		if (fx < _fbest) {
			std::copy(x.begin(), x.end(), _xbest.begin());
			_fbest = fx;
		}
	}

	// initialize work arrays
	_p = std::vector<double>(_n);
	_ws = std::vector<double>(_np);
	_wmu = std::vector<double>(4);
}

void APSOSearch::iterate() {
	updateParameters();
	updateSwarm();
	_it++;
}

multivariate_solution APSOSearch::solution(){
	return {_xbest,_fev, converged()};
}

multivariate_solution APSOSearch::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	bool converge = false;
	while (_it < _maxit && _fev < _mfev) {
		iterate();
		if (converged()) {
			converge = true;
			break;
		}
	}
	return {_xbest,_fev, converge};
}

bool APSOSearch::converged(){

	// finish when the swarm diversity becomes too small
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

	return m2 <= (_np - 1) * _tol * _tol;
}

/* =============================================================
 *
 * 				PARTICLE EVOLUTION SUBROUTINES
 *
 * =============================================================
 */
void APSOSearch::updateSwarm() {
	for (auto &p : _swarm) {
		updateParticle(p);
	}
}

void APSOSearch::updateParticle(apso_particle &particle) {

	// update the velocity and position of this particle (1)-(2)
	auto &px = particle._x;
	auto &pv = particle._v;
	for (int i = 0; i < _n; i++) {

		// compute velocity
		const double r1 = Random::get(0., 1.);
		const double r2 = Random::get(0., 1.);
		pv[i] = pv[i] * _w + _c1 * r1 * (particle._xb[i] - px[i])
				+ _c2 * r2 * (_xbest[i] - px[i]);

		// clip velocity
		const double vmax = _vdamp * (_upper[i] - _lower[i]);
		pv[i] = std::max(-vmax, std::min(pv[i], vmax));

		// update particle position
		px[i] += particle._v[i];
	}

	// correct if out of box
	if (_correct) {
		for (int i = 0; i < _n; i++) {
			px[i] = std::max(_lower[i], std::min(px[i], _upper[i]));
		}
	}

	// fitness re-evaluation and update best point so far
	particle._f = _f._f(&px[0]);
	_fev++;
	if (particle._f < particle._fb) {
		std::copy(px.begin(), px.end(), particle._xb.begin());
		particle._fb = particle._f;
	}
	if (particle._f < _fbest) {
		std::copy(px.begin(), px.end(), _xbest.begin());
		_fbest = particle._f;
	}
}

void APSOSearch::updateElitist() {

	// perturb P[d]
	std::copy(_xbest.begin(), _xbest.end(), _p.begin());
	const int d = Random::get(0, _n - 1);
	const double sigma = _smax - (_smax - _smin) * _it / _maxit;
	_p[d] += (_upper[d] - _lower[d]) * Random::get(_Z) * sigma;
	if (_correct) {
		_p[d] = std::max(_lower[d], std::min(_p[d], _upper[d]));
	}

	// if new point is a new best, then replace best point
	const double nu = _f._f(&_p[0]);
	_fev++;
	if (nu < _fbest) {
		std::copy(_p.begin(), _p.end(), _xbest.begin());
		_fbest = nu;
		return;
	}

	// if new point is not the best, replace the worst
	auto *worst = &_swarm[0];
	for (auto &p : _swarm) {
		if (p._f > worst->_f) {
			worst = &p;
		}
	}
	std::copy(_p.begin(), _p.end(), worst->_x.begin());
	worst->_f = nu;
	if (nu < worst->_fb) {
		std::copy(worst->_x.begin(), worst->_x.end(), worst->_xb.begin());
		worst->_fb = nu;
	}
}

/* =============================================================
 *
 * 				AUTOMATIC PSO PARAMETER UPDATES
 *
 * =============================================================
 */
void APSOSearch::updateParameters() {
	const double f = getf();
	const int newstate = nextState(f);
	updatec1c2(f, newstate);
	_state = newstate;
}

void APSOSearch::updatec1c2(double f, int state) {

	// update w in (10)
	_w = 1. / (1. + 1.5 * std::exp(-2.6 * f));

	// update C1 and C2 in (11)-(12)
	const double delta1 = Random::get(0.05, 0.1);
	const double delta2 = Random::get(0.05, 0.1);
	switch (state) {
	case 1: {

		// exploring
		_c1 += delta1;
		_c2 -= delta2;
		break;
	}
	case 2: {

		// exploiting
		_c1 += 0.5 * delta1;
		_c2 -= 0.5 * delta2;
		break;
	}
	case 3: {

		// converging
		_c1 += 0.5 * delta1;
		_c2 += 0.5 * delta2;

		// elitist learning
		updateElitist();
		break;
	}
	default: {

		// jumping out
		_c1 -= 0.5 * delta1;
		_c2 += 0.5 * delta2;
		break;
	}
	}
	_c1 = std::max(1.5, std::min(_c1, 2.5));
	_c2 = std::max(1.5, std::min(_c2, 2.5));

	// use equation (12)
	if (_c1 + _c2 > 4.) {
		const double fac = 4. / (_c1 + _c2);
		_c1 *= fac;
		_c2 *= fac;
	}
}

double APSOSearch::getf() {

	// calculate the distances between the particles (7)
	double dmin = std::numeric_limits<double>::infinity();
	double dmax = -std::numeric_limits<double>::infinity();
	int ibest = 0;
	for (int i = 0; i < _np; i++) {
		_ws[i] = 0.;
		const auto &xi = _swarm[i]._x;
		for (int j = 0; j < _np; j++) {
			if (j != i) {
				double dij = 0.;
				const auto &xj = _swarm[j]._x;
				for (int k = 0; k < _n; k++) {
					const double dist = xi[k] - xj[k];
					dij += dist * dist;
				}
				dij = std::sqrt(dij);
				_ws[i] += dij;
			}
		}
		_ws[i] /= (_np - 1.);

		// update the least and greatest distance
		dmax = std::max(dmax, _ws[i]);
		dmin = std::min(dmin, _ws[i]);

		// search for the best point in swarm
		if (_swarm[i]._f < _swarm[ibest]._f) {
			ibest = i;
		}
	}

	// compute the evolutionary factor in (8)
	if (dmax <= dmin) {
		return 1.;
	} else {
		return (_ws[ibest] - dmin) / (dmax - dmin);
	}
}

/* =============================================================
 *
 * 				PARAMETERS FOR STATE TRACKING
 *
 * =============================================================
 */
int APSOSearch::nextState(double f) {

	// compute the decision regions for the next state
	const double m1 = mu(f, 1);
	const double m2 = mu(f, 2);
	const double m3 = mu(f, 3);
	const double m4 = mu(f, 4);
	_wmu[0] = m1;
	_wmu[1] = m2;
	_wmu[2] = m3;
	_wmu[3] = m4;

	// at the beginning of iteration (state = 0), use hard selection
	if (_state == 0) {
		return 1 + (std::max_element(_wmu.begin(), _wmu.end()) - _wmu.begin());
	}

	// fuzzy selection with transition S1 => S2 => S3 => S4 => S1 ...
	int r;
	if (m1 > 0 && m2 > 0) {
		r = 4;
	} else if (m2 > 0 && m3 > 0) {
		r = 5;
	} else if (m1 > 0 && m4 > 0) {
		r = 6;
	} else if (m1 > 0) {
		r = 0;
	} else if (m2 > 0) {
		r = 1;
	} else if (m3 > 0) {
		r = 2;
	} else if (m4 > 0) {
		r = 3;
	} else {
		throw std::invalid_argument(
				"Error [PSO]: Invalid rule base. Please report this issue on Github.");
	}
	return _rulebase[r][_state];
}

double APSOSearch::mu(double f, int i) {

	// update the fuzzy set membership function
	switch (i) {
	case 1: {

		// exploration
		if (f >= 0. && f <= 0.4) {
			return 0.;
		} else if (f > 0.4 && f <= 0.6) {
			return 5. * f - 2.;
		} else if (f > 0.6 && f <= 0.7) {
			return 1.;
		} else if (f > 0.7 && f <= 0.8) {
			return -10. * f + 8.;
		} else {
			return 0.;
		}
		break;
	}
	case 2: {

		// exploitation
		if (f >= 0. && f <= 0.2) {
			return 0.;
		} else if (f > 0.2 && f <= 0.3) {
			return 10. * f - 2.;
		} else if (f > 0.3 && f <= 0.4) {
			return 1.;
		} else if (f > 0.4 && f <= 0.6) {
			return -5. * f + 3.;
		} else {
			return 0.;
		}
		break;
	}
	case 3: {

		// convergence
		if (f >= 0. && f <= 0.1) {
			return 1.;
		} else if (f > 0.1 && f <= 0.3) {
			return -5. * f + 1.5;
		} else {
			return 0.;
		}
		break;
	}
	case 4: {

		// jumping out
		if (f >= 0. && f <= 0.7) {
			return 0.;
		} else if (f > 0.7 && f <= 0.9) {
			return 5. * f - 3.5;
		} else {
			return 1.;
		}
		break;
	}
	default: {
		throw std::invalid_argument(
				"Error [PSO]: Invalid rule base. Please report this issue on Github.");
	}
	}
}
