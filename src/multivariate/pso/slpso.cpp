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

 [1] Li, Changhe, Shengxiang Yang, and Trung Thanh Nguyen. "A self-learning particle
 swarm optimizer for global optimization problems." IEEE Transactions on Systems,
 Man, and Cybernetics, Part B (Cybernetics) 42.3 (2011): 627-646.
 */

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>

#include "../../blas.h"
#include "../../random.hpp"

#include "slpso.h"

using Random = effolkronium::random_static;

SLPSOSearch::SLPSOSearch(int mfev, double stol, int np, double omegamin,
		double omegamax, double eta, double gamma, double vmax, double Ufmax) {
	_mfev = mfev;
	_tol = stol;
	_np = np;
	_omegamin = omegamin;
	_omegamax = omegamax;
	_eta = eta;
	_gamma = gamma;
	_vmax = vmax;
	_Ufmax = Ufmax;
	_nstrat = 4;
}

void SLPSOSearch::init(const multivariate_problem &f, const double *guess) {

	// define problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [SLPSO]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(_f._lower, _f._lower + _n);
	_upper = std::vector<double>(_f._upper, _f._upper + _n);

	// define parameters
	_mfes = 0;
	_omega = _omegamax;
	_r = std::vector<double>(_nstrat);
	_perm = std::vector<int>(_np);
	for (int p = 0; p < _np; p++) {
		_perm[p] = p;
	}
	Random::shuffle(_perm.begin(), _perm.end());

	// define swarm
	_fev = 0;
	_fabest = std::numeric_limits<double>::infinity();
	_abest = std::vector<double>(_n);
	_vdavg = std::vector<double>(_n);
	_swarm = std::vector<slpso_particle>(_np);
	for (int p = 0; p < _np; p++) {
		_swarm[p] = { };
		initializeParticle(p);
		if (_swarm[p]._fx < _fabest) {
			std::copy(_swarm[p]._x.begin(), _swarm[p]._x.end(), _abest.begin());
			_fabest = _swarm[p]._fx;
		}
	}
}

void SLPSOSearch::iterate() {

	// compute vdavg separately
	std::fill(_vdavg.begin(), _vdavg.end(), 0.);
	for (const auto &p : _swarm) {
		for (int d = 0; d < _n; d++) {
			_vdavg[d] += std::fabs(p._v[d]) / _np;
		}
	}

	// this performs one iteration of Algorithm 4
	_alpha = Random::get(0., 1.);
	int kp = 0;
	for (auto &p : _swarm) {

		// does this particle use the convergence operator?
		p._PF = p._CF;
		int i;
		if (_np * Random::get(0., 1.) < _mfes) {
			i = _nstrat - 1;
		} else {
			i = roulette(p);
		}
		p._CF = (i == _nstrat - 1);

		// continue with the particle update
		auto &k = update(i, kp, p);

		// update success probabilities
		k._G[i]++;
		if (k._fx < k._fp) {
			k._g[i]++;
			k._m = 0;
			k._p[i] += (k._fp - k._fx);
			updateAbest(k);
		} else {
			k._m++;
		}

		// update personal and abest particles
		if (k._fx < k._fpb) {
			std::copy(k._x.begin(), k._x.end(), k._pb.begin());
			k._fpb = k._fx;
			if (k._fx < _fabest) {
				std::copy(k._x.begin(), k._x.end(), _abest.begin());
				_fabest = k._fx;
			}
		}

		// update selection ratios
		if (k._m >= k._Uf) {
			updateSelectionRatios(k);
			std::fill(k._p.begin(), k._p.end(), 0.);
			std::fill(k._g.begin(), k._g.end(), 0);
			std::fill(k._G.begin(), k._G.end(), 0);
		}
		kp++;
	}
	updatePar();
}

multivariate_solution SLPSOSearch::solution(){
	return {_abest, _fev, converged()};
}

multivariate_solution SLPSOSearch::optimize(const multivariate_problem &f,
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
	return {_abest, _fev, converge};
}

bool SLPSOSearch::converged(){

	// compute standard deviation of swarm radiuses
	int count = 0;
	double mean = 0.;
	double m2 = 0.;
	for (const auto &k : _swarm) {
		const double x = dnrm2(_n, &k._x[0]);
		count++;
		const double delta = x - mean;
		mean += delta / count;
		const double delta2 = x - mean;
		m2 += delta * delta2;
	}

	return m2 <= (_np - 1) * _tol * _tol;
}

int SLPSOSearch::roulette(slpso_particle &k) {

	// select one learning operator i using roulette wheel selection
	const double s = std::accumulate(k._s.begin(), k._s.end(), 0.);
	double U = Random::get(0., s);
	for (int i = 0; i < _nstrat; i++) {
		U -= k._s[i];
		if (U <= 0.) {
			return i;
		}
	}
	return 3;
}

/* =============================================================
 *
 * 				PARTICLE EVOLUTION SUBROUTINES
 *
 * =============================================================
 */
void SLPSOSearch::initializeParticle(int index) {

	// initialize position and velocity information
	auto &k = _swarm[index];
	k._x = std::vector<double>(_n);
	k._v = std::vector<double>(_n, 0.);
	k._pb = std::vector<double>(_n);
	for (int d = 0; d < _n; d++) {
		k._x[d] = Random::get(_lower[d], _upper[d]);
		k._pb[d] = k._x[d];
	}
	k._fx = _f._f(&(k._x)[0]);
	k._fpb = k._fx;
	k._fp = k._fx;
	_fev++;

	// initialize learning parameters
	k._p = std::vector<double>(_nstrat, 0.);
	k._g = std::vector<int>(_nstrat, 0);
	k._G = std::vector<int>(_nstrat, 0);
	k._s = std::vector<double>(_nstrat, 1. / _nstrat);
	k._m = 0;
	k._CF = k._PF = false;
	updateParParticle(index);
}

SLPSOSearch::slpso_particle& SLPSOSearch::update(int i, int p,
		slpso_particle &k) {

	// this is Algorithm 1
	switch (i) {
	case 0: {

		// exploitation equation (3)
		for (int d = 0; d < _n; d++) {
			const double rkd = Random::get(0., 1.);
			k._v[d] = _omega * k._v[d] + _eta * rkd * (k._pb[d] - k._x[d]);
			const double vmax = _vmax * (_upper[d] - _lower[d]);
			k._v[d] = std::max(-vmax, std::min(k._v[d], vmax));
			k._x[d] = handleBounds(k._x[d], k._v[d], d);
		}
		k._fp = k._fx;
		k._fx = _f._f(&(k._x)[0]);
		_fev++;
		return k;
	}
	case 1: {

		// jumping out equation (4)
		for (int d = 0; d < _n; d++) {
			const double z = Random::get(_Z);
			k._x[d] = handleBounds(k._x[d], _vdavg[d] * z, d);
		}
		k._fp = k._fx;
		k._fx = _f._f(&(k._x)[0]);
		_fev++;
		return k;
	}
	case 2: {

		// exploration equation (5)
		int jp = p;
		while (jp == p) {
			jp = Random::get(0, _np - 1);
		}
		auto &j = _swarm[jp];
		if (j._fpb < k._fpb) {
			for (int d = 0; d < _n; d++) {
				const double rkd = Random::get(0., 1.);
				k._v[d] = _omega * k._v[d] + _eta * rkd * (j._pb[d] - k._x[d]);
				const double vmax = _vmax * (_upper[d] - _lower[d]);
				k._v[d] = std::max(-vmax, std::min(k._v[d], vmax));
				k._x[d] = handleBounds(k._x[d], k._v[d], d);
			}
			k._fp = k._fx;
			k._fx = _f._f(&(k._x)[0]);
			_fev++;
			return k;
		} else {
			for (int d = 0; d < _n; d++) {
				const double rkd = Random::get(0., 1.);
				j._v[d] = _omega * j._v[d] + _eta * rkd * (k._pb[d] - j._x[d]);
				const double vmax = _vmax * (_upper[d] - _lower[d]);
				j._v[d] = std::max(-vmax, std::min(j._v[d], vmax));
				j._x[d] = handleBounds(j._x[d], j._v[d], d);
			}
			j._fp = j._fx;
			j._fx = _f._f(&(j._x)[0]);
			_fev++;
			return j;
		}
	}
	case 3: {

		// convergence equation (6)
		for (int d = 0; d < _n; d++) {
			const double rkd = Random::get(0., 1.);
			k._v[d] = _omega * k._v[d] + _eta * rkd * (_abest[d] - k._x[d]);
			const double vmax = _vmax * (_upper[d] - _lower[d]);
			k._v[d] = std::max(-vmax, std::min(k._v[d], vmax));
			k._x[d] = handleBounds(k._x[d], k._v[d], d);
		}
		k._fp = k._fx;
		k._fx = _f._f(&(k._x)[0]);
		_fev++;
		return k;
	}
	default: {
		throw std::invalid_argument(
				"Error [SLPSO]: Invalid learning strategy index. Please report this issue on GitHub.");
	}
	}
}

double SLPSOSearch::handleBounds(double x, double v, int d) {

	// this is equation (11)
	const double x1 = x + v;
	if (x1 < _lower[d]) {
		return Random::get(_lower[d], x);
	} else if (x1 > _upper[d]) {
		return Random::get(x, _upper[d]);
	} else {
		return x1;
	}
}

void SLPSOSearch::updateAbest(slpso_particle &k) {

	// this is Algorithm 2
	for (int d = 0; d < _n; d++) {
		if (Random::get(0., 1.) < k._Pl) {
			const double xdold = _abest[d];
			_abest[d] = k._x[d];
			const double fnew = _f._f(&_abest[0]);
			_fev++;
			if (fnew < _fabest) {
				_fabest = fnew;
			} else {
				_abest[d] = xdold;
			}
		}
	}
}

/* =============================================================
 *
 * 				PARAMETER ADAPTATION SUBROUTINES
 *
 * =============================================================
 */
void SLPSOSearch::updatePar() {

	// this is Algorithm 5
	Random::shuffle(_perm.begin(), _perm.end());
	for (int p = 0; p < _np; p++) {
		updateParParticle(p);
	}

	// compute the number of particles that use the convergence operator by equation (16)
	const double pfev = std::max(0., std::min((1. * _fev) / _mfev, 1.));
	_mfes = static_cast<int>(_np * (1. - std::exp(-100. * std::pow(pfev, 3.))));

	// update related information of the four operators for each particle
	for (auto &k : _swarm) {
		updateLearningOpt(k);
	}

	// calculate the inertia weight by equation (17)
	_omega = _omegamax - (_omegamax - _omegamin) * pfev;
}

void SLPSOSearch::updateParParticle(int index) {
	const int k = _perm[index];

	// update U_f by equation (14)
	const double progress = std::exp(-std::pow(1.6 * k / _np, 4.));
	_swarm[index]._Uf = std::max(1., _Ufmax * progress);

	// update P_l by equation (15)
	_swarm[index]._Pl = std::max(0.05, 1. - progress);
}

void SLPSOSearch::updateLearningOpt(slpso_particle &k) {

	// this is Algorithm 3
	if (!k._CF && k._PF) {
		const double sum = std::accumulate(k._s.begin(), k._s.end() - 1, 0.);
		for (int i = 0; i < _nstrat - 1; i++) {
			k._s[i] /= sum;
		}
		k._s[_nstrat - 1] = 0.;
	}
	if (k._CF && !k._PF) {
		std::fill(k._p.begin(), k._p.end(), 0.);
		std::fill(k._g.begin(), k._g.end(), 0);
		std::fill(k._G.begin(), k._G.end(), 0);
		std::fill(k._s.begin(), k._s.end(), 1. / _nstrat);
	}
}

void SLPSOSearch::updateSelectionRatios(slpso_particle &k) {

	// update the reward values by equation (8) and (9)
	const double sump = std::accumulate(k._p.begin(), k._p.end(), 0.);
	const double smax = *std::max_element(k._s.begin(), k._s.end());
	double sumr = 0.;
	for (int i = 0; i < _nstrat; i++) {
		double cki;
		if (k._g[i] == 0 && k._s[i] >= smax) {
			cki = 0.9;
		} else {
			cki = 1.;
		}
		_r[i] = cki * k._s[i];
		if (sump > 0.) {
			_r[i] += (k._p[i] / sump) * _alpha;
		}
		if (k._G[i] > 0) {
			_r[i] += (1. * k._g[i]) / k._G[i] * (1. - _alpha);
		}
		sumr += _r[i];
	}

	// update the selection ratios by equation (10)
	for (int i = 0; i < _nstrat; i++) {
		k._s[i] = _r[i] / sumr * (1. - _nstrat * _gamma) + _gamma;
	}
}
