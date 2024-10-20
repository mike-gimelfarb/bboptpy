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

 [1] Yang, Zhenyu, Ke Tang, and Xin Yao. "Self-adaptive differential evolution
 with neighborhood search." 2008 IEEE congress on evolutionary computation
 (IEEE World Congress on Computational Intelligence). IEEE, 2008.

 [2] Gong, Wenyin, Zhihua Cai, and Yang Wang. "Repairing the crossover rate in
 adaptive differential evolution." Applied Soft Computing 15 (2014): 149-168.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include "../../blas.h"
#include "../../random.hpp"

#include "sansde.h"

using Random = effolkronium::random_static;

SaNSDESearch::SaNSDESearch(int mfev, int np, double tol, bool repaircr,
		int crref, int pupdate, int crupdate) {
	_mfev = mfev;
	_np = np;
	_tol = tol;
	_repaircr = repaircr;
	_ncrref = crref;
	_npup = pupdate;
	_ncrup = crupdate;
}

void SaNSDESearch::init(const multivariate_problem &f, const double *guess) {

	// define problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [SaNSDE]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// define parameters and memory
	_fev = 0;
	_it = 0;
	_p = 0.5;
	_fp = 0.5;
	_crrec = 0.;
	_crdeltaf = 0.;
	_crm = 0.5;
	_pns = std::vector<int>(2, 0);
	_pnf = std::vector<int>(2, 0);
	_fpns = std::vector<double>(2, 0.);
	_fpnf = std::vector<double>(2, 0.);
	_work = std::vector<double>(_n);

	// define swarm
	_swarm.clear();
	for (int i = 0; i < _np; i++) {
		std::vector<double> x(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = Random::get(_lower[j], _upper[j]);
		}
		const sansde_point part { x, _f._f(&x[0]), 0.5 };
		_swarm.push_back(std::move(part));
		_fev++;
	}
}

void SaNSDESearch::iterate() {

	// sort swarm by fitness
	std::sort(_swarm.begin(), _swarm.end(), sansde_point::compare_fitness);

	// main generation loop
	for (int i = 0; i < _np; i++) {

		// generate crossover CR
		if (_it % _ncrref == 0) {
			_swarm[i]._cr = Random::get(_Z) * 0.1 + _crm;
			_swarm[i]._cr = std::max(0., std::min(_swarm[i]._cr, 1.));
		}

		// generate mutation F
		int ifstrat;
		if (Random::get(0., 1.) < _fp) {
			ifstrat = 0;
		} else {
			ifstrat = 1;
		}
		double F = -1.;
		while (F < 0.) {
			if (ifstrat == 0) {
				F = Random::get(_Z) * 0.3 + 0.5;
			} else {
				F = sampleCauchy();
			}
			F = std::min(F, 1.);
		}

		// choose three random particles
		int irand1 = i;
		while (irand1 == i) {
			irand1 = Random::get(0, _np - 1);
		}
		int irand2 = i;
		while (irand2 == i || irand2 == irand1) {
			irand2 = Random::get(0, _np - 1);
		}
		int irand3 = i;
		while (irand3 == i || irand3 == irand1 || irand3 == irand2) {
			irand3 = Random::get(0, _np - 1);
		}

		// choose a mutation
		int istrat;
		if (Random::get(0., 1.) < _p) {
			istrat = 0;
		} else {
			istrat = 1;
		}

		// perform the mutation
		const double cr1 = mutate(&_swarm[i]._x[0], &_swarm[0]._x[0],
				&_swarm[irand1]._x[0], &_swarm[irand2]._x[0],
				&_swarm[irand3]._x[0], &_work[0], _n, F, _swarm[i]._cr, istrat);

		// bound control
		for (int j = 0; j < _n; j++) {
			if (_work[j] < _lower[j]) {
				_work[j] = (_lower[j] + _swarm[i]._x[j]) / 2.;
			} else if (_work[j] > _upper[j]) {
				_work[j] = (_upper[j] + _swarm[i]._x[j]) / 2.;
			}
		}

		// fitness selection
		const double fwork = _f._f(&_work[0]);
		_fev++;
		if (fwork < _swarm[i]._f) {

			// update strategy probabilities
			_pns[istrat]++;

			// update crossover adaptation
			const double deltaf = _swarm[i]._f - fwork;
			_crrec += cr1 * deltaf;
			_crdeltaf += deltaf;

			// update mutation adaptation
			_fpns[ifstrat] += F;

			// replace member with offspring
			_swarm[i]._f = fwork;
			std::copy(_work.begin(), _work.end(), _swarm[i]._x.begin());
		} else {
			_pnf[istrat]++;
			_fpnf[ifstrat] += F;
		}
	}

	// adapt strategy parameter
	_it++;
	if (_it > 0 && _it % _npup == 0) {
		const int pnsf0 = _pns[0] + _pnf[0];
		const int pnsf1 = _pns[1] + _pnf[1];
		_p = (1. * _pns[0] * pnsf1) / (_pns[1] * pnsf0 + _pns[0] * pnsf1);
		std::fill(_pns.begin(), _pns.end(), 0);
		std::fill(_pnf.begin(), _pnf.end(), 0);
	}

	// adapt crossover parameter
	if (_it > 0 && _it % _ncrup == 0) {
		if (_crdeltaf > 0) {
			_crm = _crrec / _crdeltaf;
		}
		_crrec = _crdeltaf = 0.;
	}

	// adapt mutation parameter
	if (_it > 0 && _it % _ncrup == 0) {
		const double fpnsf0 = _fpns[0] + _fpnf[0];
		const double fpnsf1 = _fpns[1] + _fpnf[1];
		_fp = (1. * _fpns[0] * fpnsf1)
				/ (_fpns[1] * fpnsf0 + _fpns[0] * fpnsf1);
		std::fill(_fpns.begin(), _fpns.end(), 0.);
		std::fill(_fpnf.begin(), _fpnf.end(), 0.);
	}
}

multivariate_solution SaNSDESearch::solution(){
	std::sort(_swarm.begin(), _swarm.end(), sansde_point::compare_fitness);
	return {_swarm[0]._x, _fev, converged()};
}

multivariate_solution SaNSDESearch::optimize(const multivariate_problem &f,
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
	std::sort(_swarm.begin(), _swarm.end(), sansde_point::compare_fitness);
	return {_swarm[0]._x, _fev, converge};
}

bool SaNSDESearch::converged(){

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
	return m2 <= (_np - 1) * _tol * _tol;
}

double SaNSDESearch::mutate(double *x, double *best, double *xr1, double *xr2,
		double *xr3, double *out, int n, double F, double CR, int strat) {

	// mutation
	switch (strat) {
	case 0: {

		// DE/rand/1 (exploration)
		for (int i = 0; i < n; i++) {
			out[i] = xr1[i] + F * (xr2[i] - xr3[i]);
		}
		break;
	}
	case 1: {

		// DE/current to best/2 (exploitation)
		for (int i = 0; i < n; i++) {
			out[i] = x[i] + F * (best[i] - x[i]) + F * (xr1[i] - xr2[i]);
		}
		break;
	}
	default: {
		throw std::invalid_argument(
				"Error [SaNSDE]: invalid strategy index. Please report this issue on GitHub.");
	}
	}

	// crossover
	const int irand = Random::get(0, n - 1);
	double cr1 = 0.;
	for (int i = 0; i < n; i++) {
		if (Random::get(0., 1.) <= CR || i == irand) {
			cr1 += 1.;
		} else {
			out[i] = x[i];
		}
	}
	if (_repaircr) {
		return cr1 / n;
	} else {
		return CR;
	}
}

double SaNSDESearch::sampleCauchy() {
	return std::tan(M_PI * (Random::get(0., 1.) - 0.5));
}
