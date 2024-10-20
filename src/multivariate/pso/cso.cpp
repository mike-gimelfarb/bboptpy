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

 [1] Cheng, Ran & Jin, Yaochu. (2014). A Competitive Swarm Optimizer for Large
 Scale Optimization. IEEE transactions on cybernetics. 45.
 10.1109/TCYB.2014.2322602.

 [2] Mohapatra, Prabhujit, Kedar Nath Das, and Santanu Roy. "A modified competitive
 swarm optimizer for large scale optimization problems." Applied Soft Computing 59
 (2017): 340-362.

 [3] Mohapatra, Prabhujit, Kedar Nath Das, and Santanu Roy. "Inherited competitive
 swarm optimizer for large-scale optimization problems." Harmony Search and Nature
 Inspired Optimization Algorithms. Springer, Singapore, 2019. 85-95.
 */

#include <numeric>
#include <iostream>

#include "../../blas.h"
#include "../../random.hpp"

#include "cso.h"

using Random = effolkronium::random_static;

CSOSearch::CSOSearch(int mfev, double stol, int np, int pcompete, bool ring,
		bool correct, double vmax) {
	_stol = stol;
	_mfev = mfev;
	_ring = ring;
	_correct = correct;
	_vmax = vmax;
	_pcompete = pcompete;
	if (_pcompete < 2) {
		_pcompete = 2;
		std::cerr
				<< "Warning [CSO]: particles per competition is too small - adjusted."
				<< std::endl;
	}
	_np = np;
	while (_np % _pcompete != 0) {
		_np++;
	}
}

void CSOSearch::init(const multivariate_problem &f, const double *guess) {

	// initialize function
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [CSO]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// compute parameters
	_phil = std::vector<double>(_pcompete);
	_phih = std::vector<double>(_pcompete);
	computePhi(_np);

	// initialize swarm
	_swarm.clear();
	for (int i = 0; i < _np; i++) {
		std::vector<double> x(_n);
		std::vector<double> v(_n);
		std::vector<double> mean(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = Random::get(_lower[j], _upper[j]);
			v[j] = 0.;
		}
		const cso_particle part { x, v, mean, _f._f(&x[0]) };
		_swarm.push_back(std::move(part));
	}
	_ngroup = _np / _pcompete;
	_fev = _np;

	// initialize ring topology
	if (_ring) {
		for (int i = 0; i < _np; i++) {
			const int ileft = (i - 1 + _np) % _np;
			const int iright = (i + 1) % _np;
			_swarm[i]._left = &_swarm[ileft];
			_swarm[i]._right = &_swarm[iright];
		}
	}
	_mean = std::vector<double>(_n);
	_meanw = std::vector<double>(_n);
}

void CSOSearch::iterate() {

	// update means based on neighbors topology
	if (_ring) {
		for (auto &p : _swarm) {
			for (int i = 0; i < _n; i++) {
				p._mean[i] = (p._left->_x[i] + p._x[i] + p._right->_x[i]) / 3.;
			}
		}
	} else {
		std::fill(_mean.begin(), _mean.end(), 0.);
		for (const auto &p : _swarm) {
			for (int i = 0; i < _n; i++) {
				_mean[i] += p._x[i] / _np;
			}
		}
	}

	// split m particles in the swarm into groups and sort each group by fitness
	// also compute the mean among all winners in the groups
	std::fill(_meanw.begin(), _meanw.end(), 0.);
	Random::shuffle(_swarm.begin(), _swarm.end());
	for (int p = 0; p < _np; p += _pcompete) {
		std::sort(_swarm.begin() + p, _swarm.begin() + p + _pcompete,
				cso_particle::compare);
		for (int i = 0; i < _n; i++) {
			_meanw[i] += _swarm[p]._x[i] / _ngroup;
		}
	}

	// go through each group and evolve particles
	for (int i = 0; i < _np; i += _pcompete) {
		compete(i);
	}

	// find the best and worst points
	_best = &_swarm[0];
	for (auto &p : _swarm) {
		if (p._f < _best->_f) {
			_best = &p;
		}
	}
}

multivariate_solution CSOSearch::solution(){
	return {_best->_x, _fev, converged()};
}

multivariate_solution CSOSearch::optimize(const multivariate_problem &f,
		const double *guess) {

	// initialize parameters
	init(f, guess);

	// main iteration loop over generations
	bool converge = false;
	while (_fev < _mfev) {

		// perform a single generation
		iterate();

		// test convergence in standard deviation
		if (converged()) {
			converge = true;
			break;
		}
	}
	return {_best->_x, _fev, converge};
}

bool CSOSearch::converged(){

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

	return m2 <= (_np - 1) * _stol * _stol;
}

void CSOSearch::computePhi(int m) {
	if (_pcompete == 2) {

		// validated parameters in the original paper
		if (m <= 100) {
			_phil[0] = 0.;
			_phih[0] = 0.;
		} else {
			_phil[0] = std::max(0., 0.14 * std::log(m) - 0.3);
			_phih[0] = std::max(0., 0.27 * std::log(m) - 0.51);
		}
		std::fill(_phil.begin(), _phil.end(), _phil[0]);
		std::fill(_phih.begin(), _phih.end(), _phih[0]);
	} else {

		// suggested parameters in the recent papers
		std::fill(_phil.begin(), _phil.end(), 0.);
		std::fill(_phih.begin(), _phih.end(), 0.3);
	}
}

void CSOSearch::compete(int istart) {

	// update velocity and position of the losers
	for (int p = istart + _pcompete - 1; p > istart; p--) {

		// the winner of the group is the first particle
		// the superior loser is the second particle
		// the inferior loser is the third particle, and so on
		const double phi = Random::get(_phil[p - istart], _phih[p - istart]);
		const auto &parent = _swarm[p - 1];
		auto &particle = _swarm[p];
		for (int i = 0; i < _n; i++) {

			// determine which mean to tend to
			double xmean;
			if (p == istart + 1) {

				// update for the superior loser (equation 1)
				if (_ring) {
					xmean = particle._mean[i];
				} else {
					xmean = _mean[i];
				}
			} else {

				// update for the inferior loser(s) (equation (3)
				xmean = _meanw[i];
			}

			// velocity update
			const double r1 = Random::get(0., 1.);
			const double r2 = Random::get(0., 1.);
			const double r3 = Random::get(0., 1.);
			particle._v[i] = r1 * particle._v[i]
					+ r2 * (parent._x[i] - particle._x[i])
					+ phi * r3 * (xmean - particle._x[i]);

			// clip velocity
			const double maxv = _vmax * (_upper[i] - _lower[i]);
			particle._v[i] = std::max(-maxv, std::min(particle._v[i], maxv));

			// update position
			particle._x[i] += particle._v[i];
			if (_correct) {
				particle._x[i] = std::max(_lower[i],
						std::min(particle._x[i], _upper[i]));
			}
		}

		// update the fitness of the loser
		particle._f = _f._f(&(particle._x)[0]);
		_fev++;
	}
}
