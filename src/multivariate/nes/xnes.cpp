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

 [1] Glasmachers, T., Schaul, T., Yi, S., Wierstra, D., & Schmidhuber, J.
 (2010, July). Exponential natural evolution strategies. In Proceedings of the
 12th annual conference on Genetic and evolutionary computation (pp. 393-400).
 */

#include "xnes.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include "../../blas.h"
#include "../../math_utils.h"
#include "../../random.hpp"

using Random = effolkronium::random_static;

xNES::xNES(int mfev, double tol, double a0, double etamu) {
	_tol = tol;
	_mfev = mfev;
	_a0 = a0;
	_etamu = etamu;
}

void xNES::init(const multivariate_problem &f, const double *guess) {

	// initialize domain
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [xNES]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// adaptive initialization of parameters
	_np = 4 + static_cast<int>(3. * std::log(_n));
	_etasigma = 3. * (3. + std::log(_n)) / (5. * _n * std::sqrt(_n));
	_etab = _etasigma;

	// compute utilities
	_u = std::vector<double>(_np);
	double sumu = 0.;
	for (int i = 1; i <= _np; i++) {
		const double ui = std::max(0., std::log(1 + 0.5 * _np) - std::log(i));
		_u[i - 1] = ui;
		sumu += ui;
	}
	for (int i = 0; i < _np; i++) {
		_u[i] = _u[i] / sumu - (1. / _np);
	}

	// initialize mean and covariance
	_mu = std::vector<double>(_n, 0.);
	_sigma = _a0;
	_B.clear();
	_B.resize(_n, std::vector<double>(_n, 0.));
	for (int i = 0; i < _n; i++) {
		_B[i][i] = _a0 / _sigma;
	}

	// initialize points
	_points.clear();
	for (int i = 0; i < _np; i++) {
		std::vector<double> x(_n);
		std::vector<double> z(_n);
		const xnes_point point { 0., x, z };
		_points.push_back(std::move(point));
	}
	_fev = 0;

	// initialize work memory
	_Gdelta = std::vector<double>(_n);
	_G.clear();
	_G.resize(_n, std::vector<double>(_n, 0.));
	_diagd = std::vector<double>(_n);
	_artmp = std::vector<double>(_n);
	_b.clear();
	_b.resize(_n, std::vector<double>(_n, 0.));
	_c.clear();
	_c.resize(_n, std::vector<double>(_n, 0.));
}

void xNES::iterate() {

	// sample points
	for (int i = 0; i < _np; i++) {
		for (int d = 0; d < _n; d++) {
			_points[i]._z[d] = Random::get(_Z);
		}
		for (int d = 0; d < _n; d++) {
			_points[i]._x[d] = _mu[d];
			for (int j = 0; j < _n; j++) {
				_points[i]._x[d] += _sigma * _B[d][j] * _points[i]._z[j];
			}
		}
		_points[i]._f = _f._f(&_points[i]._x[0]);
	}
	_fev += _np;

	// sort points by fitness
	std::sort(_points.begin(), _points.end(), xnes_point::compare_fitness);

	// compute G_delta
	for (int d = 0; d < _n; d++) {
		_Gdelta[d] = 0.0;
		for (int i = 0; i < _np; i++) {
			_Gdelta[d] += _u[i] * _points[i]._z[d];
		}
	}

	// compute G_M
	for (int d1 = 0; d1 < _n; d1++) {
		for (int d2 = 0; d2 < _n; d2++) {
			_G[d1][d2] = 0.0;
			for (int i = 0; i < _np; i++) {
				const double zid1 = _points[i]._z[d1];
				const double zid2 = _points[i]._z[d2];
				_G[d1][d2] += _u[i] * (zid1 * zid2 - ((d1 == d2) ? 1. : 0.));
			}
		}
	}

	// compute G_sigma
	_Gsigma = 0.0;
	for (int d = 0; d < _n; d++) {
		_Gsigma += _G[d][d] / _n;
	}

	// compute G_B
	for (int d1 = 0; d1 < _n; d1++) {
		for (int d2 = 0; d2 < _n; d2++) {
			_G[d1][d2] -= _Gsigma * ((d1 == d2) ? 1. : 0.);
		}
	}

	// update mean
	for (int d = 0; d < _n; d++) {
		for (int j = 0; j < _n; j++) {
			_mu[d] += _etamu * _sigma * _B[d][j] * _Gdelta[j];
		}
	}

	// update sigma
	_sigma *= std::exp(0.5 * _etasigma * _Gsigma);

	// update covariance
	for (int d1 = 0; d1 < _n; d1++) {
		for (int d2 = 0; d2 < _n; d2++) {
			_c[d1][d2] = 0.5 * _etab * _G[d1][d2];
		}
	}
	exponential();
	for (int d1 = 0; d1 < _n; d1++) {
		for (int d2 = 0; d2 < _n; d2++) {
			_G[d1][d2] = 0.0;
			for (int j = 0; j < _n; j++) {
				_G[d1][d2] += _B[d1][j] * _c[j][d2];
			}
		}
	}
	for (int d = 0; d < _n; d++) {
		std::copy(_G[d].begin(), _G[d].end(), _B[d].begin());
	}
}

multivariate_solution xNES::solution(){
	return {_points[0]._x, _fev, converged()};
}

multivariate_solution xNES::optimize(const multivariate_problem &f,
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
	return {_points[0]._x, _fev, converge};
}

bool xNES::converged() {
	const double fb = _points[0]._f;
	const double fw = _points[_np - 1]._f;
	if (std::abs(fb - fw) < _tol) {
		return true;
	} else {
		return false;
	}
}

// ==============================================================
//
// Computation of the matrix exponential
//
// ==============================================================

void xNES::exponential() {

	// enforce symmetry
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j <= i; j++) {
			_b[i][j] = _b[j][i] = _c[i][j];
		}
	}

	// eigenvalue decomposition
	tred2();
	tql2();

	// expC = B * exp(D) * B^T
	for (int i = 0; i < _n; i++) {
		_diagd[i] = std::exp(_diagd[i]);
	}
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j <= i; j++) {
			double sum = 0.;
			for (int k = 0; k < _n; k++) {
				sum += _b[i][k] * _diagd[k] * _b[j][k];
			}
			_c[i][j] = _c[j][i] = sum;
		}
	}
}

void xNES::tred2() {

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

void xNES::tql2() {
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

