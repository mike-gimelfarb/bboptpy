/*
 Original FORTRAN77 version by R ONeill.
 FORTRAN90 version by John Burkardt.
 C++ version by Mike Gimelfarb with changes:
 1. added adaptive parameters
 2. random starts
 3. other initialization schemes

 This code is distributed under the GNU LGPL license.

 ================================================================
 REFERENCES:

 [1] Nelder, John A.; R. Mead (1965). "A simplex method for function minimization".
 Computer Journal. 7 (4): 308–313. doi:10.1093/comjnl/7.4.308

 [2] O'Neill, R. (1971). Algorithm AS 47: Function Minimization Using a Simplex Procedure.
 Journal of the Royal Statistical Society. Series C (Applied Statistics), 20(3), 338-345.
 doi:10.2307/2346772

 [3] Gao, Fuchang & Han, Lixing. (2012). Implementing the Nelder-Mead simplex algorithm
 with adaptive parameters. Computational Optimization and Applications. 51. 259-277.
 10.1007/s10589-010-9329-3.

 [4] Mehta, V. K. "Improved Nelder–Mead algorithm in high dimensions with adaptive parameters
 based on Chebyshev spacing points." Engineering Optimization 52.10 (2020): 1814-1828.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "../../random.hpp"

#include "nelder_mead.h"

using Random = effolkronium::random_static;

NelderMead::NelderMead(int mfev, double tol, double rad0,
		simplex_initializer minit, parameter_initializer pinit, int checkev,
		double eps) {
	_mfev = mfev;
	_tol = tol;
	_rad = rad0;
	_minit = minit;
	_pinit = pinit;
	_checkev = checkev;
	_eps = eps;
}

void NelderMead::init(const multivariate_problem &f, const double *guess) {

	// problem initialization
	if (f._hasc || f._hasbbc) {
		std::cerr
				<< "Warning [NelderMead]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_start = std::vector<double>(guess, guess + _n);
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// parameters
	initParameters();

	// storage
	_p.clear();
	_p.resize(_n + 1, std::vector<double>(_n, 0.));
	_p2star = std::vector<double>(_n, 0.);
	_pbar = std::vector<double>(_n, 0.);
	_pstar = std::vector<double>(_n, 0.);
	_y = std::vector<double>(_n + 1, 0.);
	_xmin = std::vector<double>(_n, 0.);
	_step = std::vector<double>(_n);
	std::fill(_step.begin(), _step.end(), _rad);

	// Initialization.
	_icount = 0;
	_jcount = _checkev;
	_del = 1.;
	_rq = _tol * _tol * _n;
	_ynl = 0.;
}

void NelderMead::iterate() {

	// YNEWLO is, of course, the HIGHEST value???
	_conv = false;
	_ihi = std::max_element(_y.begin(), _y.end()) - _y.begin() + 1;
	_ynl = _y[_ihi - 1];

	// Calculate PBAR, the centroid of the simplex vertices
	// excepting the vertex with Y value YNEWLO.
	for (int i = 1; i <= _n; i++) {
		double sum = 0.;
		for (int k = 1; k <= _n + 1; k++) {
			sum += _p[k - 1][i - 1];
		}
		sum -= _p[_ihi - 1][i - 1];
		sum /= _n;
		_pbar[i - 1] = sum;
	}

	// Reflection through the centroid.
	for (int k = 1; k <= _n; k++) {
		_pstar[k - 1] = _pbar[k - 1]
				+ _rcoef * (_pbar[k - 1] - _p[_ihi - 1][k - 1]);
	}
	_ystar = _f._f(&_pstar[0]);
	_icount++;

	// Successful reflection, so extension.
	if (_ystar < _ylo) {

		// Expansion.
		for (int k = 1; k <= _n; k++) {
			_p2star[k - 1] = _pbar[k - 1]
					+ _ecoef * (_pstar[k - 1] - _pbar[k - 1]);
		}
		_y2star = _f._f(&_p2star[0]);
		_icount++;

		// Retain extension or contraction.
		if (_ystar < _y2star) {
			std::copy(_pstar.begin(), _pstar.end(), _p[_ihi - 1].begin());
			_y[_ihi - 1] = _ystar;
		} else {
			std::copy(_p2star.begin(), _p2star.end(), _p[_ihi - 1].begin());
			_y[_ihi - 1] = _y2star;
		}
	} else {

		// No extension.
		int l = 0;
		for (int i = 1; i <= _n + 1; i++) {
			if (_ystar < _y[i - 1]) {
				l++;
			}
		}
		if (1 < l) {

			// Copy pstar to the worst (HI) point.
			std::copy(_pstar.begin(), _pstar.end(), _p[_ihi - 1].begin());
			_y[_ihi - 1] = _ystar;
		} else if (l == 0) {

			// Contraction on the Y(IHI) side of the centroid.
			for (int k = 1; k <= _n; k++) {
				_p2star[k - 1] = _pbar[k - 1]
						+ _ccoef * (_p[_ihi - 1][k - 1] - _pbar[k - 1]);
			}
			_y2star = _f._f(&_p2star[0]);
			_icount++;

			// Contract the whole simplex.
			if (_y[_ihi - 1] < _y2star) {
				for (int j = 1; j <= _n + 1; j++) {
					for (int k = 1; k <= _n; k++) {
						_p[j - 1][k - 1] = _scoef
								* (_p[j - 1][k - 1] + _p[_ilo - 1][k - 1]);
					}
					std::copy(_p[j - 1].begin(), _p[j - 1].begin() + _n,
							_xmin.begin());
					_y[j - 1] = _f._f(&_xmin[0]);
					_icount++;
				}
				_ilo = std::min_element(_y.begin(), _y.end()) - _y.begin() + 1;
				_ylo = _y[_ilo - 1];
				_conv = false;
				return;
			} else {

				// Retain contraction.
				std::copy(_p2star.begin(), _p2star.end(), _p[_ihi - 1].begin());
				_y[_ihi - 1] = _y2star;
			}
		} else if (l == 1) {

			// Contraction on the reflection side of the centroid.
			for (int k = 1; k <= _n; k++) {
				_p2star[k - 1] = _pbar[k - 1]
						+ _ccoef * (_pstar[k - 1] - _pbar[k - 1]);
			}
			_y2star = _f._f(&_p2star[0]);
			_icount++;

			// Retain reflection?
			if (_y2star <= _ystar) {
				std::copy(_p2star.begin(), _p2star.end(), _p[_ihi - 1].begin());
				_y[_ihi - 1] = _y2star;
			} else {
				std::copy(_pstar.begin(), _pstar.end(), _p[_ihi - 1].begin());
				_y[_ihi - 1] = _y2star;
			}
		}
	}

	// Check if YLO improved.
	if (_y[_ihi - 1] < _ylo) {
		_ylo = _y[_ihi - 1];
		_ilo = _ihi;
	}
	_jcount--;
	if (0 < _jcount) {
		_conv = false;
		return;
	}

	// Check to see if minimum reached.
	if (_icount <= _mfev) {
		_jcount = _checkev;
		const double sum = std::accumulate(_y.begin(), _y.end(), 0.)
				/ (_n + 1.);
		double sumsq = 0.;
		for (int k = 1; k <= _n + 1; ++k) {
			sumsq += (_y[k - 1] - sum) * (_y[k - 1] - sum);
		}
		if (sumsq <= _rq) {
			_conv = true;
		}
	}
}

multivariate_solution NelderMead::solution(){
	return {_xmin, _icount, _conv};
}

multivariate_solution NelderMead::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	const int ifault = nelmin();
	return {_xmin, _icount, ifault == 0};
}

/* =============================================================
 *
 * 				SIMPLEX RESTART SUBROUTINES
 *
 * =============================================================
 */
int NelderMead::nelmin() {

	// Initial or restarted loop.
	while (true) {

		// Define the initial simplex
		initSimplex();
		for (int j = 0; j <= _n; j++) {
			_y[j] = _f._f(&(_p[j])[0]);
		}
		_icount += (_n + 1);

		// Find highest and lowest Y values. YNEWLO = Y(IHI) indicates
		// the vertex of the simplex to be replaced.
		_ilo = std::min_element(_y.begin(), _y.end()) - _y.begin() + 1;
		_ylo = _y[_ilo - 1];

		// Inner loop.
		while (_icount < _mfev) {
			iterate();
			if (_conv) {
				break;
			}
		}

		// Factorial tests to check that YNEWLO is a local minimum.
		std::copy(_p[_ilo - 1].begin(), _p[_ilo - 1].begin() + _n,
				_xmin.begin());
		_ynl = _y[_ilo - 1];
		if (_mfev < _icount) {
			return 2;
		}
		int ifault = 0;
		for (int i = 1; i <= _n; i++) {
			_del = _step[i - 1] * _eps;
			_xmin[i - 1] += _del;
			double z = _f._f(&_xmin[0]);
			_icount++;
			if (z < _ynl) {
				ifault = 2;
				break;
			}
			_xmin[i - 1] -= (_del + _del);
			z = _f._f(&_xmin[0]);
			_icount++;
			if (z < _ynl) {
				ifault = 2;
				break;
			}
			_xmin[i - 1] += _del;
		}
		if (ifault == 0) {
			return ifault;
		}

		// Restart the procedure.
		std::copy(_xmin.begin(), _xmin.end(), _start.begin());
		_del = _eps;
	}
}

/* =============================================================
 *
 * 				 INITIALIZATION SUBROUTINES
 *
 * =============================================================
 */
void NelderMead::initSimplex() {
	switch (_minit) {
	case coordinate_axis: {
		std::copy(_start.begin(), _start.begin() + _n, _p[_n].begin());
		for (int j = 1; j <= _n; j++) {
			const double x = _start[j - 1];
			_start[j - 1] += _step[j - 1] * _del;
			std::copy(_start.begin(), _start.begin() + _n, _p[j - 1].begin());
			_start[j - 1] = x;
		}
		break;
	}
	case spendley: {
		const double p = 1. / (_n * std::sqrt(2.))
				* (_n - 1. + std::sqrt(_n - 1.));
		const double q = 1. / (_n * std::sqrt(2.)) * (std::sqrt(_n + 1.) - 1.);
		std::copy(_start.begin(), _start.begin() + _n, _p[_n].begin());
		for (int i = 1; i <= _n; i++) {
			for (int j = 1; j <= _n; j++) {
				if (i == j) {
					_p[i - 1][j - 1] = _start[j - 1] + _step[j - 1] * _del * p;
				} else {
					_p[i - 1][j - 1] = _start[j - 1] + _step[j - 1] * _del * q;
				}
			}
		}
		break;
	}
	case pfeffer: {
		const double du = 0.05, dz = 0.0075;
		std::copy(_start.begin(), _start.begin() + _n, _p[_n].begin());
		for (int i = 1; i <= _n; i++) {
			for (int j = 1; j <= _n; j++) {
				if (i == j) {
					if (_start[j - 1] == 0.) {
						_p[i - 1][j - 1] = dz;
					} else {
						_p[i - 1][j - 1] = _start[j - 1] * (1. + du);
					}
				} else {
					_p[i - 1][j - 1] = _start[j - 1];
				}
			}
		}
		break;
	}
	case random: {
		std::copy(_start.begin(), _start.begin() + _n, _p[_n].begin());
		for (int i = 1; i <= _n; i++) {
			for (int j = 1; j <= _n; j++) {
				_p[i - 1][j - 1] = Random::get(_lower[j - 1], _upper[j - 1]);
			}
		}
	}
	}
}

void NelderMead::initParameters() {
	switch (_pinit) {
	case original: {
		_ccoef = 0.5;
		_ecoef = 2.0;
		_rcoef = 1.0;
		_scoef = 0.5;
		break;
	}
	case gao2010: {
		_ccoef = 0.75 - 0.5 / _n;
		_ecoef = 1. + 2. / _n;
		_rcoef = 1.;
		_scoef = 1. - 1. / _n;
		break;
	}
	case mehta2019_crude: {
		_ccoef = 1. + std::cos((_n + 3. + (_n % 2)) * M_PI / (2. * _n));
		_ecoef = 1. + std::cos((_n - 3. - (_n % 2)) * M_PI / (2. * _n));
		_rcoef = 1. + std::cos((_n - 1. - (_n % 2)) * M_PI / (2. * _n));
		_scoef = 1. + std::cos((_n + 1. + (_n % 2)) * M_PI / (2. * _n));
		break;
	}
	case mehta2019_refined: {
		const int nc = 2 * (9 + (_n - 1) / 5);
		_ccoef = 1. + std::cos((nc + 5.) * M_PI / (2. * nc));
		_ecoef = 1. + std::cos((nc - 3.) * M_PI / (2. * nc));
		_rcoef = 1. + std::cos((nc - 1.) * M_PI / (2. * nc));
		_scoef = 1. + std::cos((nc + 3.) * M_PI / (2. * nc));
		break;
	}
	}
}
