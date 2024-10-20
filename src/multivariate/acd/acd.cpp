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

 [1] Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Adaptive coordinate
 descent." Proceedings of the 13th annual conference on Genetic and evolutionary
 computation. 2011.

 [2] Hansen, Nikolaus. "Adaptive encoding: How to render search coordinate system
 invariant." International Conference on Parallel Problem Solving from Nature.
 Springer, Berlin, Heidelberg, 2008.
 */

#include <cmath>
#include <iostream>
#include <numeric>

#include "../../blas.h"
#include "../../math_utils.h"
#include "../../random.hpp"

#include "acd.h"

using Random = effolkronium::random_static;

ACD::ACD(int mfev, double ftol, double xtol, double ksucc, double kunsucc) {
	_mfev = mfev;
	_ftol = ftol;
	_xtol = xtol;
	_ksucc = ksucc;
	_kunsucc = kunsucc;
}

void ACD::init(const multivariate_problem &f, const double *guess) {

	// initialize problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [ACD]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// parameters
	_c1 = 0.5 / _n;
	_cmu = 0.5 / _n;
	_cp = 1. / std::sqrt(1. * _n);
	_updateperiod = 1;
	_ix = 0;
	_it = 0;
	_itae = 0;
	_fev = 0;
	_improved = false;
	_fbest = std::numeric_limits<double>::infinity();

	// initialize best and sigma
	_xbest = std::vector<double>(_n);
	_sigma = std::vector<double>(_n);
	for (int i = 0; i < _n; i++) {
		_xbest[i] = Random::get(_lower[i], _upper[i]);
		_sigma[i] = (_upper[i] - _lower[i]) / 4.;
	}

	// generate population
	_points.clear();
	_order = std::vector<int>(2 * _n);
	for (int i = 0; i < 2 * _n; i++) {
		std::vector<double> x(_n);
		const point pt { x, 0. };
		_points.push_back(std::move(pt));
		_order[i] = i;
	}

	// parameters for the adaptive encoding
	_artmp = std::vector<double>(_n);
	_x1 = std::vector<double>(_n);
	_x2 = std::vector<double>(_n);
	_weights = std::vector<double>(_n, 1. / _n);
	_p = std::vector<double>(_n, 0.);
	_m = std::vector<double>(_n);
	_mold = std::vector<double>(_n);
	_diagd = std::vector<double>(_n, 1.);
	_b.clear();
	_b.resize(_n, std::vector<double>(_n, 0.));
	_invB.clear();
	_invB.resize(_n, std::vector<double>(_n, 0.));
	_c.clear();
	_c.resize(_n, std::vector<double>(_n, 0.));
	for (int i = 0; i < _n; i++) {
		_b[i][i] = 1.;
		_invB[i][i] = 1.;
		_c[i][i] = 1.;
	}

	// convergence monitor
	_convergeperiod = 10 + static_cast<int>(20 * std::pow(1. * _n, 1.5));
	_fhist = std::vector<double>(_convergeperiod);
}

void ACD::iterate() {

	// generate two candidate solutions
	for (int i = 0; i < _n; i++) {
		const double dx = _sigma[_ix] * _b[i][_ix];
		_x1[i] = std::max(_lower[i], std::min(_xbest[i] - dx, _upper[i]));
		_x2[i] = std::max(_lower[i], std::min(_xbest[i] + dx, _upper[i]));

	}
	const double f1 = _f._f(&_x1[0]);
	const double f2 = _f._f(&_x2[0]);
	_fev += 2;

	// update xmean and fbest
	const bool success = f1 < _fbest || f2 < _fbest;
	if (f1 < _fbest) {
		std::copy(_x1.begin(), _x1.end(), _xbest.begin());
		_fbest = f1;
	}
	if (f2 < _fbest) {
		std::copy(_x2.begin(), _x2.end(), _xbest.begin());
		_fbest = f2;
	}

	// update convergence history
	_fhist[_it % _convergeperiod] = _fbest;

	// adapt the step size sigma
	if (success) {
		_sigma[_ix] *= _ksucc;
		_improved = true;
	} else {
		_sigma[_ix] *= _kunsucc;
	}

	// update archive
	std::copy(_x1.begin(), _x1.end(), _points[2 * _ix]._x.begin());
	_points[2 * _ix]._f = f1;
	std::copy(_x2.begin(), _x2.end(), _points[2 * _ix + 1]._x.begin());
	_points[2 * _ix + 1]._f = f2;

	// update adaptive encoding
	if (_improved && _ix == _n - 1) {

		// find the best individuals
		std::sort(_order.begin(), _order.end(),
				[&](const int i, const int j) -> bool {
					return _points[i]._f < _points[j]._f;
				});

		// update encoding
		updateAE();
		_improved = false;
	}
	_ix = (_ix + 1) % _n;
	_it++;
}

multivariate_solution ACD::solution(){
	return {_xbest, _fev, converged()};
}

multivariate_solution ACD::optimize(const multivariate_problem &f,
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
	return {_xbest, _fev, converge};
}

/* =============================================================
 *
 * 				CONVERGENCE TEST SUBROUTINES
 *
 * =============================================================
 */
bool ACD::converged() {

	// convergence in history of f values
	if (_it > _convergeperiod) {
		const double f0 = _fhist[(_it - 1 + _convergeperiod) % _convergeperiod];
		const double f1 = _fhist[_it % _convergeperiod];
		if (std::fabs(f1 - f0) < _ftol) {
			return true;
		}
	}

	// convergence based on change in x values
	double dxmax = 0.;
	for (int ix = 0; ix < _n; ix++) {
		for (int i = 0; i < _n; i++) {
			const double dx = _sigma[ix] * _b[i][ix];
			dxmax = std::max(dxmax, std::fabs(dx));
		}
	}
	if (dxmax < _xtol) {
		return true;
	}

	return false;
}

/* =============================================================
 *
 * 				ADAPTIVE ENCODING SUBROUTINES
 *
 * =============================================================
 */
void ACD::updateAE() {

	// first iteration
	_itae++;
	if (_itae == 1) {
		std::fill(_m.begin(), _m.end(), 0.);
		for (int i = 0; i < _n; i++) {
			for (int d = 0; d < _n; d++) {
				_m[d] += _points[_order[i]]._x[d] * _weights[i];
			}
		}
		return;
	}

	// subsequent iteration - compute the new mean
	std::copy(_m.begin(), _m.end(), _mold.begin());
	std::fill(_m.begin(), _m.end(), 0.);
	for (int i = 0; i < _n; i++) {
		for (int d = 0; d < _n; d++) {
			_m[d] += _points[_order[i]]._x[d] * _weights[i];
		}
	}

	// compute p_c
	for (int i = 0; i < _n; i++) {
		_artmp[i] = 0.;
		for (int j = 0; j < _n; j++) {
			_artmp[i] += _invB[i][j] * (_m[j] - _mold[j]);
		}
	}
	double denom = std::inner_product(_artmp.begin(), _artmp.end(),
			_artmp.begin(), 0.);
	if (denom <= 0.) {
		for (int i = 0; i < _n; i++) {
			_p[i] *= (1. - _cp);
		}
	} else {
		const double factor = std::sqrt(_cp * (2. - _cp) * _n / denom);
		for (int i = 0; i < _n; i++) {
			_p[i] = (1. - _cp) * _p[i] + factor * (_m[i] - _mold[i]);
		}
	}

	// compute C
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j < _n; j++) {
			_c[i][j] = (1. - _c1) * _c[i][j] + _c1 * _p[i] * _p[j];
		}
	}

	// eigenvalue decomposition, B==normalized eigenvectors
	if (true) {

		// tridiagonal
		for (int i = 0; i < _n; i++) {
			for (int j = 0; j <= i; j++) {
				_b[i][j] = _b[j][i] = _c[i][j];
			}
		}
		tred2();
		tql2();

		// limit condition number of C
		if (_diagd[0] <= 0.) {
			for (int i = 0; i < _n; i++) {
				_diagd[i] = std::max(_diagd[i], 0.);
			}
			const double shift = _diagd[_n - 1] / 1e14;
			for (int i = 0; i < _n; i++) {
				_c[i][i] += shift;
				_diagd[i] += shift;
			}
		}
		if (_diagd[_n - 1] > 1e14 * _diagd[0]) {
			const double shift = _diagd[_n - 1] / 1e14 - _diagd[0];
			for (int i = 0; i < _n; i++) {
				_c[i][i] += shift;
				_diagd[i] += shift;
			}
		}

		// take square root of eigenvalues
		for (int i = 0; i < _n; i++) {
			_diagd[i] = std::sqrt(_diagd[i]);
		}

		// update invB and B
		for (int i = 0; i < _n; i++) {
			for (int j = 0; j < _n; j++) {
				_invB[i][j] = _b[j][i] / _diagd[i];
			}
		}
		for (int i = 0; i < _n; i++) {
			for (int j = 0; j < _n; j++) {
				_b[i][j] *= _diagd[j];
			}
		}
	}
}

void ACD::tred2() {

	// This is derived from the Algol procedures tred2 by
	// Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	// Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	// Fortran subroutine in EISPACK.
	std::copy(_b[_n - 1].begin(), _b[_n - 1].end(), _diagd.begin());

	// Householder reduction to tridiagonal form.
	for (int i = _n - 1; i > 0; i--) {

		// Scale to avoid under/overflow.
		double scale = 0.;
		double h = 0.;
		for (int k = 0; k < i; k++) {
			scale += std::abs(_diagd[k]);
		}
		if (scale == 0.) {
			_artmp[i] = _diagd[i - 1];
			for (int j = 0; j < i; j++) {
				_diagd[j] = _b[i - 1][j];
				_b[i][j] = _b[j][i] = 0.;
			}
		} else {

			// Generate Householder vector.
			for (int k = 0; k < i; k++) {
				_diagd[k] /= scale;
				h += _diagd[k] * _diagd[k];
			}
			double f = _diagd[i - 1];
			double g = std::sqrt(h);
			if (f > 0) {
				g = -g;
			}
			_artmp[i] = scale * g;
			h = h - f * g;
			_diagd[i - 1] = f - g;
			std::fill(_artmp.begin(), _artmp.begin() + i, 0.);

			// Apply similarity transformation to remaining columns.
			for (int j = 0; j < i; j++) {
				f = _diagd[j];
				_b[j][i] = f;
				g = _artmp[j] + _b[j][j] * f;
				for (int k = j + 1; k <= i - 1; k++) {
					g += _b[k][j] * _diagd[k];
					_artmp[k] += _b[k][j] * f;
				}
				_artmp[j] = g;
			}
			dscalm(i, 1. / h, &_artmp[0], 1);
			f = std::inner_product(_artmp.begin(), _artmp.begin() + i,
					_diagd.begin(), 0.);
			const double hh = f / (h + h);
			daxpym(i, -hh, &_diagd[0], 1, &_artmp[0], 1);
			for (int j = 0; j < i; j++) {
				f = _diagd[j];
				g = _artmp[j];
				for (int k = j; k <= i - 1; k++) {
					_b[k][j] -= (f * _artmp[k] + g * _diagd[k]);
				}
				_diagd[j] = _b[i - 1][j];
				_b[i][j] = 0.;
			}
		}
		_diagd[i] = h;
	}

	// Accumulate transformations.
	for (int i = 0; i < _n - 1; i++) {
		_b[_n - 1][i] = _b[i][i];
		_b[i][i] = 1.;
		const double h = _diagd[i + 1];
		if (h != 0.) {
			for (int k = 0; k <= i; k++) {
				_diagd[k] = _b[k][i + 1] / h;
			}
			for (int j = 0; j <= i; j++) {
				double g = 0.;
				for (int k = 0; k <= i; k++) {
					g += _b[k][i + 1] * _b[k][j];
				}
				for (int k = 0; k <= i; k++) {
					_b[k][j] -= g * _diagd[k];
				}
			}
		}
		for (int k = 0; k <= i; k++) {
			_b[k][i + 1] = 0.;
		}
	}
	std::copy(_b[_n - 1].begin(), _b[_n - 1].end(), _diagd.begin());
	std::fill(_b[_n - 1].begin(), _b[_n - 1].end(), 0.);
	_b[_n - 1][_n - 1] = 1.;
	_artmp[0] = 0.;
}

void ACD::tql2() {
	for (int i = 1; i < _n; i++) {
		_artmp[i - 1] = _artmp[i];
	}
	_artmp[_n - 1] = 0.;
	double f = 0., tst1 = 0.;
	double eps = std::pow(2., -52.);
	for (int l = 0; l < _n; l++) {

		// Find small subdiagonal element
		tst1 = std::max(tst1, std::abs(_diagd[l]) + std::abs(_artmp[l]));
		int m = l;
		for (m = l; m < _n; m++) {
			if (std::abs(_artmp[m]) <= eps * tst1) {
				break;
			}
		}
		if (m >= _n) {
			break;
		}

		// If m == l, d[l] is an eigenvalue, otherwise, iterate.
		if (m > l) {
			do {

				// Compute implicit shift
				double g = _diagd[l];
				double p = (_diagd[l + 1] - g) / (2. * _artmp[l]);
				double r = hypot(p, 1.);
				r = sign(r, p);
				_diagd[l] = _artmp[l] / (p + r);
				_diagd[l + 1] = _artmp[l] * (p + r);
				double dl1 = _diagd[l + 1];
				double h = g - _diagd[l];
				for (int i = l + 2; i < _n; i++) {
					_diagd[i] -= h;
				}
				f += h;

				// Implicit QL transformation.
				p = _diagd[m];
				double c = 1., c2 = c, c3 = c;
				double el1 = _artmp[l + 1];
				double s = 0., s2 = 0.;
				for (int i = m - 1; i >= l; i--) {
					c3 = c2;
					c2 = c;
					s2 = s;
					g = c * _artmp[i];
					h = c * p;
					r = hypot(p, _artmp[i]);
					_artmp[i + 1] = s * r;
					s = _artmp[i] / r;
					c = p / r;
					p = c * _diagd[i] - s * g;
					_diagd[i + 1] = h + s * (c * g + s * _diagd[i]);

					// Accumulate transformation.
					for (int k = 0; k < _n; k++) {
						h = _b[k][i + 1];
						_b[k][i + 1] = s * _b[k][i] + c * h;
						_b[k][i] = c * _b[k][i] - s * h;
					}
				}
				p = -s * s2 * c3 * el1 * _artmp[l] / dl1;
				_artmp[l] = s * p;
				_diagd[l] = c * p;

				// Check for convergence.
			} while (std::abs(_artmp[l]) > eps * tst1);
		}
		_diagd[l] += f;
		_artmp[l] = 0.;
	}

	// Sort eigenvalues and corresponding vectors.
	for (int i = 0; i < _n - 1; i++) {
		int k = i;
		double p = _diagd[i];
		for (int j = i + 1; j < _n; j++) {
			if (_diagd[j] < p) {
				k = j;
				p = _diagd[j];
			}
		}
		if (k != i) {
			_diagd[k] = _diagd[i];
			_diagd[i] = p;
			for (int j = 0; j < _n; j++) {
				p = _b[j][i];
				_b[j][i] = _b[j][k];
				_b[j][k] = p;
			}
		}
	}
}
