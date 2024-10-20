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

 [1] Rosenbrock, HoHo. "An automatic method for finding the greatest or least
 value of a function." The Computer Journal 3.3 (1960): 175-184.

 [2] Swann, W. H. (1964). Report on the Development of a new Direct Search
 Method of Optimisation, Imperial Chemical Industries Ltd., Central Instrument
 Laboratory Research Note 64/3.

 [3] Palmer, J. R. "An improved procedure for orthogonalising the search
 vectors in Rosenbrock's and Swann's direct search optimisation methods." The
 Computer Journal 12.1 (1969): 69-71.

 [4] Box, M. J.; Davies, D.; Swann, W. H. (1969). Non-Linear optimisation
 Techniques. Oliver & Boyd.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include "../../blas.h"

#include "rosenbrock.h"

Rosenbrock::Rosenbrock(int mfev, double tol, double step0, double decf) {
	_tol = tol;
	_step0 = step0;
	_rho = decf;
	_mfev = mfev;
}

void Rosenbrock::init(const multivariate_problem &f, const double *guess) {

	// prepare problem
	if (f._hasc || f._hasbbc) {
		std::cerr
				<< "Warning [Rosenbrock]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);
	_x0 = std::vector<double>(guess, guess + _n);

	// prepare memory
	_x1 = std::vector<double>(_n);
	_v.clear();
	_v.resize(_n + 2, std::vector<double>(_n, 0.));
	_vold.clear();
	_vold.resize(_n + 2, std::vector<double>(_n, 0.));
	_x.clear();
	_x.resize(_n + 2, std::vector<double>(_n, 0.));
	_d = std::vector<double>(_n + 2);
	_temp = std::vector<double>(_n + 1);
	_fs = std::vector<double>(4);
	_wx = std::vector<double>(_n);

	// first step
	std::copy(_x0.begin(), _x0.end(), _x[0].begin());
	_stepi = _step0;
	for (int j = 1; j <= _n; j++) {
		_v[j][j - 1] = 1.;
	}
	_fev = 0;
	_i = 1;
	_ierr = -1;
	_wa = 0.;
}

void Rosenbrock::iterate() {

	// PERFORM A LINE SEARCH USING LAGRANGE QUADRATIC INTERPOLATION
	// SUGGESTED BY DAVIES, SWANN AND CAMPEY. SEE SWANN (1964) OR
	// BOX ET AL (1969)
	_wa = _stepi;
	const int err = lineSearch(_f._f, _n, &_x[_i - 1][0], _wa, &_v[_i][0],
			&_x[_i][0], &_wx[0], &_fs[0], _fev, _mfev);
	_d[_i] = _wa;

	// REACHED MAXIMUM NUMBER OF EVALUATIONS
	if (err != 0) {
		_ierr = 1;
		return;
	}

	// WARM-UP PERIOD - DON'T DO ANYTHING UNTIL WE HAVE N POINTS
	if (_i < _n) {
		_i++;
		return;
	}

	// MAIN LOOP
	if (_i == _n) {

		// EVENTUALLY DO ONE MORE LINE SEARCH...
		daxpy1(_n, -1., &_x[0][0], 1, &_x[_n][0], 1, &_temp[0], 1);
		const double zn = dnrm2(_n, &_temp[0]);
		if (zn > 0.) {
			dscal1(_n, 1. / zn, &_temp[0], 1, &_v[_n + 1][0], 1);
			_i = _n + 1;
			return;
		} else {
			std::copy(_x[_n].begin(), _x[_n].end(), _x[_n + 1].begin());
			_d[_n + 1] = 0.;
		}
	} else {

		// COMPUTE THE ERROR
		double dxn = 0.;
		for (int j = 1; j <= _n; j++) {
			const double tmp = _x[_n + 1][j - 1] - _x[0][j - 1];
			dxn += (tmp * tmp);
		}
		dxn = std::sqrt(dxn);

		// CHECK APPROPRIATENESS OF STEP LENGTH
		if (dxn >= _stepi) {

			// COPY THE OLD BASIS VECTORS
			for (int ii = 1; ii <= _n; ii++) {
				std::copy(_v[ii].begin(), _v[ii].end(), _vold[ii].begin());
			}

			// COMPUTE THE QUANTITIES SUM(I:N) D(I)^2 AND PLACE THEM
			// INTO AN AUXILIARY ARRAY
			for (int j = _n; j >= 1; j--) {
				if (j == _n) {
					_temp[j] = _d[j] * _d[j];
				} else {
					_temp[j] = _temp[j + 1] + (_d[j] * _d[j]);
				}
			}

			// PERFORM ORTHOGONALIZATION USING A MODIFICATION
			// OF THE GRAHAM-SCHMIDT ORTHOGONALIZATION PROCEDURE BY
			// J. PALMER (1969)
			for (int ii = 1; ii <= _n; ii++) {
				if (_temp[ii] <= 0.) {
					continue;
				}
				if (ii == 1) {
					for (int j = 1; j <= _n; j++) {
						_v[ii][j - 1] = 0.;
						for (int jj = 1; jj <= _n; jj++) {
							_v[ii][j - 1] += (_d[jj] * _vold[jj][j - 1]);
						}
						_v[ii][j - 1] /= std::sqrt(_temp[ii]);
					}
				} else {
					for (int j = 1; j <= _n; j++) {
						_v[ii][j - 1] = 0.;
						for (int jj = ii; jj <= _n; jj++) {
							_v[ii][j - 1] += (_d[jj] * _vold[jj][j - 1]);
						}
						_v[ii][j - 1] *= _d[ii - 1];
						_v[ii][j - 1] -= (_vold[ii - 1][j - 1] * _temp[ii]);
						_v[ii][j - 1] /= std::sqrt(_temp[ii] * _temp[ii - 1]);
					}
				}
			}
			_d[1] = _d[_n + 1];
			std::copy(_x[_n].begin(), _x[_n].end(), _x[0].begin());
			std::copy(_x[_n + 1].begin(), _x[_n + 1].end(), _x[1].begin());
			_i = 2;
			return;
		}
	}

	// TERMINATION CRITERION
	_stepi *= _rho;
	if (_stepi <= _tol) {
		std::copy(_x[_n + 1].begin(), _x[_n + 1].end(), _x1.begin());
		_ierr = 0;
		return;
	} else {
		std::copy(_x[_n + 1].begin(), _x[_n + 1].end(), _x[0].begin());
		_i = 1;
	}

	// REACHED MAXIMUM NUMBER OF EVALUATIONS
	if (_fev >= _mfev) {
		_ierr = 1;
		return;
	}
}

multivariate_solution Rosenbrock::solution(){
	return {_x1, _fev, _ierr == 0};
}

multivariate_solution Rosenbrock::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	while (true) {
		iterate();
		if (_ierr >= 0) {
			return {_x1, _fev, _ierr == 0};
		}
	}
}

int Rosenbrock::lineSearch(const multivariate &f, int n, double *pos, double &s,
		double *v, double *x, double *x0, double *fs, int &fev, int mfev) {

	// INITIALIZATION
	std::fill(fs, fs + 4, 0.);
	std::copy(pos, pos + n, x0);
	double fx0 = f(&x0[0]);
	fev++;

	// STEP FORWARD
	daxpy1(n, s, &v[0], 1, &x0[0], 1, x, 1);
	double fx = f(x);
	fev++;

	if (fx > fx0) {

		// STEP BACKWARD
		daxpym(n, -2. * s, v, 1, x, 1);
		s = -s;
		fx = f(x);
		fev++;
		if (fx > fx0) {
			goto l4;
		}
	}

	// FURTHER STEPS
	do {
		s *= 2.;
		std::copy(x, x + n, x0);
		fx0 = fx;
		daxpy1(n, s, v, 1, &x0[0], 1, x, 1);
		fx = f(x);
		fev++;
		if (fev > mfev) {
			return 1;
		}
	} while (fx <= fx0 && std::fabs(s) < 1e30);

	// PREPARE INTERPOLATION
	s /= 2.;
	daxpy1(n, s, v, 1, &x0[0], 1, x, 1);

	l4:
	// GENERATE THE FOUR POSSIBLE INTERPOLATION POINTS AND THE VALUES
	daxpy1(n, -s, v, 1, &x0[0], 1, x, 1);
	fs[0] = f(x);
	std::copy(x0, x0 + n, x);
	fs[1] = f(x);
	daxpy1(n, s, v, 1, &x0[0], 1, x, 1);
	fs[2] = f(x);
	daxpy1(n, 2. * s, v, 1, &x0[0], 1, x, 1);
	fs[3] = f(x);
	fev += 4;

	// IGNORE THE POINT THAT IS FURTHEST FROM THE MINIMUM OF THE FOUR
	// POINTS. FOR THE REMAINING THREE POINTS, COMPUTE THE LAGRANGE
	// QUADRATIC INTERPOLATION BY FITTING A PARABOLA THROUGH THE POINTS
	// AND COMPUTE THE MINIMUM
	double stepf;
	const int imin = std::min_element(fs, fs + 4) - fs;
	if (imin == 1) {
		const double num = s * (fs[0] - fs[2]);
		const double den = 2. * (fs[0] - 2. * fs[1] + fs[2]);
		stepf = 0.;
		if (std::fabs(den) > 0.) {
			stepf += (num / den);
		}
	} else if (imin == 2) {
		const double num = s * (fs[1] - fs[3]);
		const double den = 2. * (fs[1] - 2. * fs[2] + fs[3]);
		stepf = s;
		if (std::fabs(den) > 0.) {
			stepf += (num / den);
		}
	} else {

		// COULD NOT FIND THE BEST INTERPOLATION POINT SO RETURN THE MIN
		if (imin == 0) {
			stepf = -s;
		} else {
			stepf = 2. * s;
		}
		daxpy1(n, stepf, v, 1, &x0[0], 1, x, 1);
		s = stepf;
		return 0;
	}

	// COMPUTE THE POINT AND FUNCTION VALUE AT THE INTERPOLATED STEP
	daxpy1(n, stepf, v, 1, &x0[0], 1, x, 1);
	fx = f(x);
	fev++;

	// IF THIS FUNCTION VALUE EXCEEDS F2, THEN RESTORE THE POINT BACK
	// TO THE MIDPOINT OF THE INTERPOLATION INTERVAL
	if ((imin == 1 && fx > fs[1]) || (imin == 2 && fx > fs[2])) {
		if (imin == 1) {
			stepf = 0.;
		} else {
			stepf = s;
		}
		daxpy1(n, stepf, v, 1, &x0[0], 1, x, 1);
	}
	s = stepf;
	return 0;
}
