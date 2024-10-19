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

 [1] Luo, Kaiping. "A Novel Self-Adaptive Harmony Search Algorithm."
 Journal of Applied Mathematics 2013.1 (2013): 653749.
 */

#include <algorithm>
#include <limits>
#include <numeric>
#include <iostream>

#include "../../random.hpp"

#include "nshs.h"

using Random = effolkronium::random_static;

NSHS::NSHS(int mfev, int hms, double fstdmin) {
	_hms = hms;
	_mfev = mfev;
	_fstdmin = fstdmin;
}

void NSHS::init(const multivariate_problem &f, const double *guess) {

	// problem initialization
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [NSHS]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// harmony memory
	_hm.clear();
	for (int i = 0; i < _hms; i++) {
		std::vector<double> x(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = Random::get(_lower[j], _upper[j]);
		}
		const harmony har { x, _f._f(&x[0]) };
		_hm.push_back(std::move(har));
	}

	// recalculate std of memory
	calculate_std();

	// recalculate best point
	_best = &*std::min_element(_hm.begin(), _hm.end(),
			harmony::compare_fitness);

	// temporary
	std::vector<double> x(_n);
	_temp = harmony { x, 0. };

	// other memory
	_fev = _hms;
	_it = 0;
	_mit = _mfev - _hms;

	// initialize parameters
	_hmcr = 1.0 - 1.0 / (_n + 1);
	_tunerange = 0.0;
	for (int d = 0; d < _n; d++) {
		_tunerange = std::max(_tunerange, (_upper[d] - _lower[d]) / 2);
	}
}

void NSHS::iterate() {
	generate_harmony();
	replace();
	_it++;
}

multivariate_solution NSHS::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	while (_fev < _mfev) {
		iterate();
	}
	return {_best->_x, _fev, false};
}

void NSHS::generate_harmony() {

	// generate a new value for the i-th variable
	for (int i = 0; i < _n; i++) {

		// generate a new value for the variable
		if (Random::get(0.0, 1.0) < _hmcr) {
			const int j = Random::get(0, _hms - 1);
			_temp._x[i] = _hm[j]._x[i];
		} else {
			if (_fstd > _fstdmin) {
				_temp._x[i] = Random::get(_lower[i], _upper[i]);
			} else {
				double minj = _upper[i];
				double maxj = _lower[i];
				for (int j = 0; j < _hms; j++) {
					minj = std::min(minj, _hm[j]._x[i]);
					maxj = std::max(maxj, _hm[j]._x[i]);
				}
				_temp._x[i] = Random::get(minj, maxj);
			}
		}

		// adjust the new value
		if (_fstd > _fstdmin) {
			const double bwi = ((_upper[i] - _lower[i]) / _tunerange)
					* (1.0 - (1. * _it) / _mit);
			_temp._x[i] += Random::get(-bwi, bwi);
		} else {
			_temp._x[i] += Random::get(-_fstdmin, _fstdmin);
		}

		// ensure new trial harmony is in the valid bounds
		_temp._x[i] = std::min(_temp._x[i], _upper[i]);
		_temp._x[i] = std::max(_temp._x[i], _lower[i]);
	}

	// assess the merit of the new harmony vector
	_temp._f = _f._f(&_temp._x[0]);
	++_fev;
}

void NSHS::replace() {

	// find which harmony is the worst and replace it
	auto &worst = *std::max_element(_hm.begin(), _hm.end(),
			harmony::compare_fitness);
	if (_temp._f < worst._f) {
		std::copy(_temp._x.begin(), _temp._x.end(), (worst._x).begin());
		worst._f = _temp._f;
	}

	// recalculate std of memory
	calculate_std();

	// recalculate best point
	_best = &*std::min_element(_hm.begin(), _hm.end(),
			harmony::compare_fitness);
}

void NSHS::calculate_std() {
	double _fmean = 0.0;
	double _fm2 = 0.0;
	for (int i = 0; i < _hms; i++) {
		const double delta = _hm[i]._f - _fmean;
		_fmean += delta / (i + 1);
		_fm2 += delta * (_hm[i]._f - _fmean);
	}
	_fstd = std::sqrt(_fm2 / _hms);
}

