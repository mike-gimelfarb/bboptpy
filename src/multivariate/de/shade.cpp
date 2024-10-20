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

 [1] Tanabe, Ryoji, and Alex Fukunaga. "Success-history based parameter
 adaptation for differential evolution." 2013 IEEE congress on evolutionary
 computation. IEEE, 2013.

 [2] Tanabe, Ryoji, and Alex S. Fukunaga. "Improving the search performance of
 SHADE using linear population size reduction." 2014 IEEE congress on
 evolutionary computation (CEC). IEEE, 2014.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include "../../blas.h"
#include "../../random.hpp"

#include "shade.h"

using Random = effolkronium::random_static;

ShadeSearch::ShadeSearch(int mfev, int npinit, double tol, bool archive,
		bool repaircr, int h, int npmin) {
	_mfev = mfev;
	_tol = tol;
	_archive = archive;
	_repaircr = repaircr;
	_h = h;
	_npinit = npinit;
	_npmin = npmin;
}

void ShadeSearch::init(const multivariate_problem &f, const double *guess) {

	// define problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [SHADE]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// define parameters and memory
	_fev = 0;
	_MCR = std::vector<double>(_h, 0.5);
	_MF = std::vector<double>(_h, 0.5);
	_work = std::vector<double>(_n);
	_arch.clear();
	_SCR.clear();
	_SF.clear();
	_w.clear();
	_k = 1;
	_np = _npinit;

	// define swarm
	_swarm.clear();
	for (int i = 0; i < _np; i++) {
		std::vector<double> x(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = Random::get(_lower[j], _upper[j]);
		}
		const point part { x, _f._f(&x[0]) };
		_swarm.push_back(std::move(part));
		_fev++;
	}

	// sort swarm by fitness
	std::sort(_swarm.begin(), _swarm.end(), point::compare_fitness);
}

void ShadeSearch::iterate() {

	// main generation loop
	_SCR.clear();
	_SF.clear();
	_w.clear();
	for (int i = 0; i < _np; i++) {

		// sample a crossover parameter
		const int ri = Random::get(0, _h - 1);
		const double CRi = std::max(0.,
				std::min(Random::get(_Z) * 0.1 + _MCR[ri], 1.));

		// sample a mutation parameter
		double Fi = std::min(1., sampleCauchy() * 0.1 + _MF[ri]);
		while (Fi <= 0) {
			Fi = std::min(1., sampleCauchy() * 0.1 + _MF[ri]);
		}

		// sample a greediness parameter
		const double pi = Random::get(std::min(2. / _n, 0.2), 0.2);

		// choose randomly one of the best particles
		const int nelite = std::max(1, static_cast<int>(pi * _np));
		const int ibest = Random::get(0, nelite - 1);

		// choose another two random particles
		int irand1 = i;
		while (irand1 == i) {
			irand1 = Random::get(0, _np - 1);
		}
		int irand2 = i;
		int larch = static_cast<int>(_arch.size());
		while (irand2 == i || irand2 == irand1) {
			irand2 = Random::get(0, _np + larch - 1);
		}

		// perform the mutation
		double cr1;
		if (irand2 >= _np) {

			// x2 sampled from archive
			cr1 = mutate(&_swarm[i]._x[0], &_swarm[ibest]._x[0],
					&_swarm[irand1]._x[0], &_arch[irand2 - _np][0], &_work[0],
					_n, Fi, CRi);
		} else {

			// x2 sampled from population
			cr1 = mutate(&_swarm[i]._x[0], &_swarm[ibest]._x[0],
					&_swarm[irand1]._x[0], &_swarm[irand2]._x[0], &_work[0], _n,
					Fi, CRi);
		}

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
		if (fwork <= _swarm[i]._f) {

			// archive management
			if (_archive && fwork < _swarm[i]._f) {
				if (larch >= _np) {
					const int irand = Random::get(0, _np - 1);
					std::copy(_swarm[i]._x.begin(), _swarm[i]._x.end(),
							_arch[irand].begin());
				} else {
					_arch.push_back(_swarm[i]._x);
				}
			}

			// memory management
			if (fwork < _swarm[i]._f) {
				_SCR.push_back(cr1);
				_SF.push_back(Fi);
				_w.push_back(_swarm[i]._f - fwork);
			}

			// replacement
			_swarm[i]._f = fwork;
			std::copy(_work.begin(), _work.end(), _swarm[i]._x.begin());
		}
	}

	// memory update
	const int lenS = static_cast<int>(_SCR.size());
	if (lenS > 0) {

		// compute mean CR and mean F
		double meanCRnum = 0.0;
		double meanCRden = 0.0;
		double meanFnum = 0.0;
		double meanFden = 0.0;
		for (int i = 0; i < lenS; i++) {
			meanCRnum += _w[i] * _SCR[i];
			meanCRden += _w[i];
			meanFnum += _w[i] * _SF[i] * _SF[i];
			meanFden += _w[i] * _SF[i];
		}
		const double meanCR = meanCRnum / meanCRden;
		const double meanF = meanFnum / meanFden;

		// replace in memory
		_MCR[_k - 1] = meanCR;
		_MF[_k - 1] = meanF;
		_k++;
		if (_k > _h) {
			_k = 1;
		}
	}

	// sort swarm by fitness
	std::sort(_swarm.begin(), _swarm.end(), point::compare_fitness);

	// population size adjustment
	const int npnew = std::round(
			(_npmin - _npinit) * ((1. * _fev) / _mfev) + _npinit);
	if (npnew < _np) {
		for (int i = 0; i < _np - npnew; i++) {
			_swarm.pop_back();
		}
		_np = npnew;
	}

	// archive size adjustment
	if (_archive) {
		int larch = static_cast<int>(_arch.size());
		while (larch > npnew) {
			const int irand = Random::get(0, larch - 1);
			_arch.erase(_arch.begin() + irand);
			--larch;
		}
	}
}

multivariate_solution ShadeSearch::solution() {
	std::sort(_swarm.begin(), _swarm.end(), point::compare_fitness);
	return {_swarm[0]._x, _fev, converged()};
}

multivariate_solution ShadeSearch::optimize(const multivariate_problem &f,
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

bool ShadeSearch::converged() {

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

double ShadeSearch::mutate(double *x, double *best, double *xr1, double *xr2,
		double *out, int n, double F, double CR) {
	const int irand = Random::get(0, n - 1);
	double cr1 = 0.;
	for (int i = 0; i < n; i++) {
		if (i == irand || Random::get(0., 1.) < CR) {
			out[i] = x[i] + F * (best[i] - x[i]) + F * (xr1[i] - xr2[i]);
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

double ShadeSearch::sampleCauchy() {
	return std::tan(M_PI * (Random::get(0., 1.) - 0.5));
}
