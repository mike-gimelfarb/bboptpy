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

 [1] Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: Self-adaptive differential
 evolution with fast and reliable convergence performance." 2007 IEEE congress
 on evolutionary computation. IEEE, 2007.

 [2] Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: adaptive differential
 evolution with optional external archive." IEEE Transactions on evolutionary
 computation 13.5 (2009): 945-958.

 [3] Li, Jie, et al. "Power mean based crossover rate adaptive differential
 evolution." International Conference on Artificial Intelligence and Computational
 Intelligence. Springer, Berlin, Heidelberg, 2011.

 [4] Gong, Wenyin, Zhihua Cai, and Yang Wang. "Repairing the crossover rate
 in adaptive differential evolution." Applied Soft Computing 15 (2014): 149-168.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include "../../blas.h"
#include "../../random.hpp"

#include "jade.h"

using Random = effolkronium::random_static;

JadeSearch::JadeSearch(int mfev, int np, double tol, bool archive,
		bool repaircr, double pelite, double cdamp, double sigma) {
	_mfev = mfev;
	_np = np;
	_tol = tol;
	_archive = archive;
	_repaircr = repaircr;
	_pelite = pelite;
	_c = cdamp;
	_sigma = sigma;
}

void JadeSearch::init(const multivariate_problem &f, const double *guess) {

	// define problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [JADE]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// define parameters and memory
	_fev = 0;
	_mucr = 0.5;
	_muf = 0.5;
	_work = std::vector<double>(_n);
	_scr = std::vector<double>(_np);
	_sf = std::vector<double>(_np);
	_arch.clear();

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
}

void JadeSearch::iterate() {

	// sort swarm by fitness
	std::sort(_swarm.begin(), _swarm.end(), point::compare_fitness);

	// main generation loop
	int nsucc = 0;
	for (int i = 0; i < _np; i++) {

		// generate crossover CR
		double CR = Random::get(_Z) * 0.1 + _mucr;
		CR = std::max(0., std::min(CR, 1.));

		// generate mutation F
		double F = -1.;
		while (F < 0.) {
			F = sampleCauchy() * 0.1 + _muf;
			F = std::min(F, 1.);
		}

		// choose randomly one of the best particles
		const int nelite = std::max(1, static_cast<int>(_pelite * _np));
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
					_n, F, CR);
		} else {

			// x2 sampled from population
			cr1 = mutate(&_swarm[i]._x[0], &_swarm[ibest]._x[0],
					&_swarm[irand1]._x[0], &_swarm[irand2]._x[0], &_work[0], _n,
					F, CR);
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
			if (_archive) {
				if (larch >= _np) {
					const int irand = Random::get(0, _np - 1);
					std::copy(_swarm[i]._x.begin(), _swarm[i]._x.end(),
							_arch[irand].begin());
				} else {
					_arch.push_back(_swarm[i]._x);
				}
			}

			// replacement
			_swarm[i]._f = fwork;
			std::copy(_work.begin(), _work.end(), _swarm[i]._x.begin());

			// memory for parameter adaptation
			_scr[nsucc] = cr1;
			_sf[nsucc] = F;
			nsucc++;
		}
	}

	// update mean CR
	double meancr;
	if (nsucc > 0) {
		if (std(&_scr[0], nsucc) > _sigma) {
			meancr = std::sqrt(meanPow(&_scr[0], nsucc, 2));
		} else {
			meancr = meanPow(&_scr[0], nsucc, 1);
		}
	} else {
		meancr = 0.;
	}
	_mucr = (1. - _c) * _mucr + _c * meancr;

	// update mean F
	double meanf;
	if (nsucc > 0) {
		meanf = meanPow(&_sf[0], nsucc, 2) / meanPow(&_sf[0], nsucc, 1);
	} else {
		meanf = 0.;
	}
	_muf = (1. - _c) * _muf + _c * meanf;
}

multivariate_solution JadeSearch::solution() {
	std::sort(_swarm.begin(), _swarm.end(), point::compare_fitness);
	return {_swarm[0]._x, _fev, converged()};
}

multivariate_solution JadeSearch::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	bool converge = false;
	while (_fev < _mfev) {
		iterate();
		if (converged()){
			converge = true;
			break;
		}
	}
	std::sort(_swarm.begin(), _swarm.end(), point::compare_fitness);
	return {_swarm[0]._x, _fev, converge};
}

bool JadeSearch::converged() {

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

double JadeSearch::mutate(double *x, double *best, double *xr1, double *xr2,
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

double JadeSearch::meanPow(double *values, int n, int p) {
	double mean = 0.;
	for (int i = 0; i < n; i++) {
		const double v = values[i];
		if (p == 1) {
			mean += (v - mean) / (i + 1.);
		} else {
			mean += (v * v - mean) / (i + 1.);
		}
	}
	return mean;
}

double JadeSearch::std(double *values, int n) {
	double mean = 0.;
	double m2 = 0.;
	for (int i = 0; i < n; i++) {
		const double v = values[i];
		const double delta = v - mean;
		mean += delta / (i + 1.);
		const double delta2 = v - mean;
		m2 += delta * delta2;
	}
	return std::sqrt(m2 / n);
}

double JadeSearch::sampleCauchy() {
	return std::tan(M_PI * (Random::get(0., 1.) - 0.5));
}
