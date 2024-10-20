/*
 Copyright (c) 1999, 2000, 2001 North Carolina State University
 Copyright (c) 2020 Mike Gimelfarb - Converted to C++ version
 This program is distributed under the MIT License (see
 http://www.opensource.org/licenses/mit-license.php):

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ================================================================
 REFERENCES:

 Jones, Donald R., Cary D. Perttunen, and Bruce E. Stuckman. "Lipschitzian
 optimization without the Lipschitz constant." Journal of optimization Theory
 and Applications 79.1 (1993): 157-181.

 [2] Gablonsky, Joerg M., and Carl T. Kelley. "A locally-biased form of the
 DIRECT algorithm." Journal of Global Optimization 21.1 (2001): 27-37.
 */

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "directl.h"

Directl::Directl(int mfev, double volper, double sigmaper, double eps,
		int method) {
	_mfev = mfev;
	_method = method;
	_eps = eps;
	_volper = volper;
	_sigmaper = sigmaper;
}

void Directl::init(const multivariate_problem &f, const double *guess) {

	// initialize problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [DIRECT]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);
	_fglobal = -1e100;
	_fglper = 0.;
	_jones = _method;
	_maxdiv = 24000;
	_maxdeep = 4800;
	_lmaxdim = 64;
	_maxf = _mfev - 21;

	_l = std::vector<double>(_n);
	_u = std::vector<double>(_n);
	std::copy(_lower.begin(), _lower.end(), _l.begin());
	std::copy(_upper.begin(), _upper.end(), _u.begin());

	// Set parameters.
	_cheat = 0;
	_kmax = 1e10;
	_mdeep = _maxdeep;

	// Write the header of the logfile.
	dirHeader(_n, _eps, _maxf, &_l[0], &_u[0], _mfev, _ierror, _epsfix,
			_iepschange);
	if (_ierror < 0) {
		return;
	}

	// If the known global minimum is equal 0, we cannot divide by it.
	if (_fglobal == 0.) {
		_divfactor = 1.;
	} else {
		_divfactor = std::fabs(_fglobal);
	}

	// Save the budget given by the user.
	_oldmaxf = _mfev;
	_inc = 0;

	// Initialiase the lists.
	_anchor = std::vector<int>(_maxdeep + 2);
	_point = std::vector<int>(_mfev);
	_farr = std::vector<double>(_mfev * 2);
	dirInitList(&_anchor[0], _free, &_point[0], &_farr[0], _mfev, _maxdeep);

	// Call the routine to initialise the mapping of x from the n-dimensional
	// unit cube to the hypercube given by u and l
	_oops = dirPreprc(&_u[0], &_l[0], _n, &_l[0], &_u[0]);
	if (_oops > 0) {
		_ierror = -3;
		throw std::invalid_argument(
				"Error [DIRECT]: bad bounding box provided.");
		return;
	}
	_tstart = 2;

	// Initialise the algorithm DIRECT.
	// Added variable to keep track of the maximum value found.
	_c = std::vector<double>(_mfev * _n);
	_length = std::vector<int>(_mfev * _n);
	_arrayi = std::vector<int>(_n);
	_list2 = std::vector<int>(_n * 2);
	_w = std::vector<double>(_n);
	_x = std::vector<double>(_n);
	_thirds = std::vector<double>(1 + _maxdeep);
	_levels = std::vector<double>(1 + _maxdeep);
	_S = std::vector<int>(_maxdiv * 2);

	dirInit(&_farr[0], _f._f, &_c[0], &_length[0], _actdeep, &_point[0],
			&_anchor[0], _free, &_arrayi[0], _maxi, &_list2[0], &_w[0], &_x[0],
			&_l[0], &_u[0], _fmin, _minpos, &_thirds[0], &_levels[0], _mfev,
			_maxdeep, _n, _fmax, _ifeas, _iifeas, _ierror);

	// Added error checking.
	if (_ierror < 0) {
		throw std::invalid_argument("Error [DIRECT]: initialization failed.");
		if (_ierror == -4) {
			return;
		}
		if (_ierror == -5) {
			return;
		}
	}
	_numfunc = 1 + _maxi + _maxi;
	_actmaxdeep = 1;
	_oldpos = 0;
	_tstart = 2;
}

void Directl::iterate() {

	// Choose the sample points. The indices of the sample points are stored
	// in the list S.
	_actdeep = _actmaxdeep;
	dirChoose(&_anchor[0], &_S[0], _maxdeep, &_farr[0], _fmin, _eps,
			&_levels[0], _maxpos, &_length[0], _mfev, _maxdeep, _maxdiv, _n,
			_cheat, _kmax, _ifeas);

	// Add other hyperrectangles to S, which have the same level and the same
	// function value at the center as the ones found above (that are stored
	// in S). This is only done if we use the original DIRECT algorithm.
	if (_method == 0) {
		dirDoubleInsert(&_anchor[0], &_S[0], _maxpos, &_point[0], &_farr[0],
				_mfev, _maxdiv, _ierror);
		if (_ierror == -6) {
			return;
		}
	}
	_oldpos = _minpos;

	// Initialise the number of sample points in this outer loop.
	_newtosample = 0;
	for (int j = 1; j <= _maxpos; j++) {
		_actdeep = _S[(2 - 1) * _maxdiv + (j - 1)];

		// If the actual index is a point to sample, do it.
		if (_S[j - 1] > 0) {

			// JG 09/24/00 Calculate the value delta used for sampling points.
			_actdeepdiv = dirGetMaxDeep(_S[j - 1], &_length[0], _mfev, _n);
			_delta = _thirds[_actdeepdiv + 1];
			_actdeep = _S[(2 - 1) * _maxdiv + (j - 1)];

			// If the current dept of division is only one under the maximal allowed
			// dept, stop the computation.
			if (_actdeep + 1 >= _mdeep) {
				_ierror = 6;
				return;
			}
			_actmaxdeep = std::max(_actdeep, _actmaxdeep);
			_help = _S[j - 1];
			if (!(_anchor[_actdeep + 1] == _help)) {
				_pos1 = _anchor[_actdeep + 1];
				while (!(_point[_pos1 - 1] == _help)) {
					_pos1 = _point[_pos1 - 1];
				}
				_point[_pos1 - 1] = _point[_help - 1];
			} else {
				_anchor[_actdeep + 1] = _point[_help - 1];
			}
			if (_actdeep < 0) {
				_actdeep = _farr[_help - 1];
			}

			// Get the Directions in which to decrease the intervall-length.
			dirGetI(&_length[0], _help, &_arrayi[0], _maxi, _n, _mfev);

			// Sample the function. To do this, we first calculate the points where
			// we need to sample the function. After checking for errors, we then do
			// the actual evaluation of the function, again followed by checking for errors
			dirSamplePoints(&_c[0], &_arrayi[0], _delta, _help, _start,
					&_length[0], _free, _maxi, &_point[0], _n, _mfev, _oops);
			if (_oops > 0) {
				_ierror = -4;
				return;
			}
			_newtosample += _maxi;

			// JG 01/22/01 Added variable to keep track of the maximum value found.
			dirSampleF(&_c[0], &_arrayi[0], _start, &_length[0], &_farr[0],
					_maxi, &_point[0], _f._f, &_x[0], &_l[0], _fmin, _minpos,
					&_u[0], _n, _mfev, _fmax, _ifeas, _iifeas);
			if (_oops > 0) {
				_ierror = -5;
				return;
			}

			// Divide the intervalls.
			dirDivide(_start, _actdeepdiv, &_length[0], &_point[0], &_arrayi[0],
					_help, &_list2[0], &_w[0], _maxi, &_farr[0], _mfev,
					_maxdeep, _n);

			// Insert the new intervalls into the list (sorted).
			dirInsertList(_start, &_anchor[0], &_point[0], &_farr[0], _maxi,
					&_length[0], _mfev, _maxdeep, _n, _help);

			// Increase the number of function evaluations.
			_numfunc += _maxi + _maxi;
		}
	}

	// JG 01/22/01 Calculate the index for the hyperrectangle at which
	// fmin is assumed. We then calculate the volume of this
	// hyperrectangle and store it in delta.
	_ierror = _jones;
	_jones = 0;
	_actdeepdiv = dirGetLevel(_minpos, &_length[0], _mfev, _n);
	_jones = _ierror;

	// JG 07/16/01 Use precalculated values to calculate volume.
	_delta = _thirds[_actdeepdiv] * 100;
	if (_delta <= _volper) {
		_ierror = 4;
		return;
	}

	// JG 01/23/01 Calculate the measure for the hyperrectangle at which
	// fmin is assumed. If this measure is smaller then sigmaper, we stop DIRECT.
	_actdeepdiv = dirGetLevel(_minpos, &_length[0], _mfev, _n);
	_delta = _levels[_actdeepdiv];
	if (_delta <= _sigmaper) {
		_ierror = 5;
		return;
	}

	// If the best found function value is within fglper of the (known)
	// global minimum value, terminate. This only makes sense if this optimal
	// value is known, that is, in test problems.
	if (100. * (_fmin - _fglobal) / _divfactor <= _fglper) {
		_ierror = 3;
		return;
	}

	// Find out if there are infeasible points which are near feasible ones
	if (_iifeas > 0) {
		dirReplaceInf(_free, &_farr[0], &_c[0], &_thirds[0], &_length[0],
				&_anchor[0], &_point[0], &_u[0], &_l[0], _mfev, _maxdeep, _n,
				_fmax);
	}

	// If iepschange = 1, we use the epsilon change formula from Jones.
	if (_iepschange == 1) {
		_eps = std::max(1e-4 * std::fabs(_fmin), _epsfix);
	}

	// If no feasible point has been found yet, set the maximum number of
	// function evaluations to the number of evaluations already done plus
	// the budget given by the user.
	// If the budget has already be increased, increase it again. If a
	// feasible point has been found, remark that and reset flag. No further
	// increase is needed.
	if (_inc == 1) {
		_maxf = _numfunc + _oldmaxf;
		if (_iifeas == 0) {
			_inc = 0;
		}
	}

	// Check if the number of function evaluations done is larger than the
	// allocated budget. If this is the case, check if a feasible point was
	// found. If this is a case, terminate. If no feasible point was found,
	// increase the budget and set flag increase.
	if (_numfunc > _maxf) {
		if (_iifeas == 0) {
			_ierror = 1;
			return;
		} else {
			_inc = 1;
			_maxf = _numfunc + _oldmaxf;
		}
	}
}

multivariate_solution Directl::solution(){
	for (int i = 1; i <= _n; i++) {
		_x[i - 1] = _c[(i - 1) * _mfev + (_minpos - 1)] * _l[i - 1]
				+ _l[i - 1] * _u[i - 1];
	}
	return {_x, _numfunc, _ierror > 0};
}

multivariate_solution Directl::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	while (_numfunc < _mfev && _ierror == 0) {
		iterate();
	}
	for (int i = 1; i <= _n; i++) {
		_x[i - 1] = _c[(i - 1) * _mfev + (_minpos - 1)] * _l[i - 1]
				+ _l[i - 1] * _u[i - 1];
	}
	return {_x, _numfunc, _ierror > 0};
}

void Directl::dirInit(double *f, const multivariate &fcn, double *c,
		int *length, int &actdeep, int *point, int *anchor, int &free,
		int *arrayi, int &maxi, int *list2, double *w, double *x, double *l,
		double *u, double &fmin, int &minpos, double *thirds, double *levels,
		int maxfunc, int maxdeep, int n, double &fmax, int &ifeas, int &iifeas,
		int &ierr) {
	fmin = 1e20;

	if (_jones == 0) {

		// JG 09/15/00 If Jones way of characterising rectangles is used
		// initialise thirds to reflect this.
		for (int j = 0; j <= n - 1; j++) {
			w[j + 1 - 1] = 0.5 * std::sqrt(n - j + j / 9.);
		}
		double help2 = 1.;
		for (int i = 1; i <= maxdeep / n; i++) {
			for (int j = 0; j <= n - 1; j++) {
				levels[(i - 1) * n + j] = w[j + 1 - 1] / help2;
			}
			help2 *= 3.;
		}
	} else {

		// JG 09/15/00 Initialiase levels to contain 1/j
		double help2 = 3.;
		for (int i = 1; i <= maxdeep; i++) {
			levels[i] = 1. / help2;
			help2 *= 3.;
		}
		levels[0] = 1.;
	}

	double help2 = 3.;
	for (int i = 1; i <= maxdeep; i++) {
		thirds[i] = 1. / help2;
		help2 *= 3.;
	}
	thirds[0] = 1.;
	for (int i = 1; i <= n; i++) {
		c[(i - 1) * maxfunc + (1 - 1)] = 0.5;
		x[i - 1] = 0.5;
		length[(i - 1) * maxfunc + (1 - 1)] = 0;
	}
	f[0] = dirInfcn(fcn, x, l, u, n);
	f[(2 - 1) * maxfunc + (1 - 1)] = 0;
	iifeas = 0;
	fmax = f[0];
	if (f[(2 - 1) * maxfunc + (1 - 1)] > 0) {
		f[0] = 1e6;
		fmax = f[0];
		ifeas = 1;
	} else {
		ifeas = 0;
	}

	fmin = f[0];
	minpos = 1;
	actdeep = 2;
	point[0] = 0;
	free = 2;
	double delta = thirds[1];
	dirGetI(length, 1, arrayi, maxi, n, maxfunc);
	int nnew = free;
	int oops;
	dirSamplePoints(c, arrayi, delta, 1, nnew, length, free, maxi, point, n,
			maxfunc, oops);

	// JG 01/23/01 Added error checking.
	if (oops > 0) {
		ierr = -4;
		return;
	}

	// JG 01/22/01 	Added variable to keep track of the maximum value found.
	// 				Added variable to keep track if feasible point was found.
	dirSampleF(c, arrayi, nnew, length, f, maxi, point, fcn, x, l, fmin, minpos,
			u, n, maxfunc, fmax, ifeas, iifeas);

	// JG 01/23/01 Added error checking.
	if (oops > 0) {
		ierr = -5;
		return;
	}
	dirDivide(nnew, 0, length, point, arrayi, 1, list2, w, maxi, f, maxfunc,
			maxdeep, n);
	dirInsertList(nnew, anchor, point, f, maxi, length, maxfunc, maxdeep, n, 1);
}

void Directl::dirHeader(int n, double &eps, int maxf, double *lower,
		double *upper, int maxfunc, int &ierror, double &epsfix,
		int &iepschange) {
	ierror = 0;

	// JG 01/13/01 Added check for epsilon.
	if (eps < 0) {
		iepschange = 1;
		epsfix = -eps;
		eps = -eps;
	} else {
		iepschange = 0;
		epsfix = 1e100;
	}

	for (int i = 1; i <= n; i++) {
		if (upper[i - 1] <= lower[i - 1]) {
			ierror = -1;
		}
	}
	if (ierror == -1) {
		throw std::invalid_argument("Error [DIRECT]: invalid bounds provided.");
	}

	// If there are to many function evaluations or to many iteration, note
	// this and set the error flag accordingly. Note: If more than one error
	// occurred, we give out an extra message.
	if (maxf + 20 > maxfunc) {
		ierror = -2;
		throw std::invalid_argument("Error [DIRECT]: budget is too small.");
	}
}

void Directl::dirSampleF(double *c, int *arrayi, int nnew, int *length,
		double *f, int maxi, int *point, const multivariate &fcn, double *x,
		double *l, double &fmin, int &minpos, double *u, int n, int maxfunc,
		double &fmax, int &ifeas, int &iifeas) {

	// Set the pointer to the first function to be evaluated,
	// store this position also in helppoint.
	int pos = nnew;
	int helpp = pos;

	// Iterate over all points, where the function should be evaluated
	for (int j = 1; j <= maxi + maxi; j++) {

		// Copy the position into the helparrayy x.
		for (int i = 1; i <= n; i++) {
			x[i - 1] = c[(i - 1) * maxfunc + (pos - 1)];
		}

		// Call the function.
		f[pos - 1] = dirInfcn(fcn, x, l, u, n);

		// Remember IF an infeasible point has been found.
		iifeas = std::max(iifeas, 0);
		if (0 == 0) {

			// IF the function evaluation was O.K., set the flag in
			// f(pos,2). Also mark that a feasible point has been found.
			f[(2 - 1) * maxfunc + (pos - 1)] = 0.;
			ifeas = 0;

			// JG 01/22/01 Added variable to keep track of the maximum value found.
			fmax = std::max(f[pos - 1], fmax);
		}
		if (0 >= 1) {

			// IF the function could not be evaluated at the given point
			// set flag to mark this (f(pos,2) and store the maximum
			// box-sidelength in f(pos,1).
			f[(2 - 1) * maxfunc + (pos - 1)] = 2.;
			f[pos - 1] = fmax;
		}

		//  IF the function could not be evaluated due to a failure in
		// the setup, mark this.
		if (0 == -1) {
			f[(2 - 1) * maxfunc + (pos - 1)] = -1.;
		}

		// Set the position to the next point, at which the function
		// should be evaluated.
		pos = point[pos - 1];
	}
	pos = helpp;

	// Iterate over all evaluated points and see, IF the minimal
	// value of the function has changed.  IF this has happEND,
	// store the minimal value and its position in the array.
	for (int j = 1; j <= maxi + maxi; j++) {
		if (f[pos - 1] < fmin && f[(2 - 1) * maxfunc + (pos - 1)] == 0) {
			fmin = f[pos - 1];
			minpos = pos;
		}
		pos = point[pos - 1];
	}
}

void Directl::dirChoose(int *anchor, int *S, int actdeep, double *f,
		double fmin, double eps, double *thirds, int &maxpos, int *length,
		int maxfunc, int maxdeep, int maxdiv, int n, int cheat, double kmax,
		int ifeas) {
	const double maxlower = 1e20;
	int i, j, k, i_, j_;
	double help2, helpl, helpg;
	int novalue, novaluedeep;

	helpl = maxlower;
	helpg = 0.;
	k = 1;
	if (ifeas >= 1) {
		for (j = 0; j <= actdeep; j++) {
			if (anchor[j + 1] > 0) {
				S[k - 1] = anchor[j + 1];
				S[(2 - 1) * maxdiv + (k - 1)] = dirGetLevel(S[k - 1], length,
						maxfunc, n);
				break;
			}
		}
		k++;
		maxpos = 1;
		return;
	} else {
		for (j = 0; j <= actdeep; j++) {
			if (anchor[j + 1] > 0) {
				S[k - 1] = anchor[j + 1];
				S[(2 - 1) * maxdiv + (k - 1)] = dirGetLevel(S[k - 1], length,
						maxfunc, n);
				k++;
			}
		}
	}
	novalue = 0;
	if (anchor[0] > 0) {
		novalue = anchor[0];
		novaluedeep = dirGetLevel(novalue, length, maxfunc, n);
	}
	maxpos = k - 1;
	for (j = k - 1; j <= maxdeep; j++) {
		S[k - 1] = 0;
	}
	for (j = maxpos; j >= 1; j--) {
		helpl = maxlower;
		helpg = 0.;
		j_ = S[j - 1];
		for (i = 1; i <= j - 1; i++) {
			i_ = S[i - 1];

			// JG 07/16/01 Changed IF statement into two to prevent run-time errors
			if (i_ > 0 && !(i == j)) {
				if (f[(2 - 1) * maxfunc + (i_ - 1)] <= 1.) {
					help2 = thirds[S[(2 - 1) * maxdiv + (i - 1)]]
							- thirds[S[(2 - 1) * maxdiv + (j - 1)]];
					help2 = (f[i_ - 1] - f[j_ - 1]) / help2;
					if (help2 <= 0.) {
						S[j - 1] = 0;
						goto l40;
					}
					if (help2 < helpl) {
						helpl = help2;
					}
				}
			}
		}
		for (i = j + 1; i <= maxpos; i++) {
			i_ = S[i - 1];

			// JG 07/16/01 Changed IF statement into two to prevent run-time errors
			if (i_ > 0 && !(i == j)) {
				if (f[(2 - 1) * maxfunc + (i_ - 1)] <= 1.) {
					help2 = thirds[S[(2 - 1) * maxdiv + (i - 1)]]
							- thirds[S[(2 - 1) * maxdiv + (j - 1)]];
					help2 = (f[i_ - 1] - f[j_ - 1]) / help2;
					if (help2 <= 0.) {
						S[j - 1] = 0;
						goto l40;
					}
					if (help2 > helpg) {
						helpg = help2;
					}
				}
			}
		}
		if (helpl > maxlower && helpg > 0) {
			helpl = helpg;
			helpg -= 1.;
		}
		if (helpg <= helpl) {
			if (cheat == 1 && helpl > kmax) {
				helpl = kmax;
			}
			if (f[j_ - 1] - helpl * thirds[S[(2 - 1) * maxdiv + (j - 1)]]
					> (fmin - eps * std::fabs(fmin))) {
				S[j - 1] = 0;
				goto l40;
			}
		} else {
			S[j - 1] = 0;
			goto l40;
		}
		l40: ;
	}
	if (novalue > 0) {
		maxpos++;
		S[maxpos - 1] = novalue;
		S[(2 - 1) * maxdiv + (maxpos - 1)] = novaluedeep;
	}
}

void Directl::dirDoubleInsert(int *anchor, int *S, int maxpos, int *point,
		double *f, int maxfunc, int maxdiv, int &ierror) {

	// JG 07/16/01 Added flag to prevent run time-errors on some systems.
	int oldmaxpos = maxpos;
	for (int i = 1; i <= oldmaxpos; i++) {
		if (S[i - 1] > 0) {
			int actdeep = S[(2 - 1) * maxdiv + (i - 1)];
			int help = anchor[actdeep + 1];
			int pos = point[help - 1];
			int iflag = 0;

			// JG 07/16/01 Added flag to prevent run time-errors on some systems.
			while (pos > 0 && iflag == 0) {
				if (f[pos - 1] - f[help - 1] <= 1e-13) {
					if (maxpos < maxdiv) {
						maxpos++;
						S[maxpos - 1] = pos;
						S[(2 - 1) * maxdiv + (maxpos - 1)] = actdeep;
						pos = point[pos - 1];
					} else {

						// JG 07/16/01 Maximum number of elements possible in S has been reached!
						ierror = -6;
						return;
					}
				} else {
					iflag = 1;
				}
			}
		}
	}
}

void Directl::dirReplaceInf(int free, double *f, double *c, double *thirds,
		int *length, int *anchor, int *point, double *c1, double *c2,
		int maxfunc, int maxdeep, int n, double fmax) {
	double *a, *b, *x;

	// allocation
	a = new double[_lmaxdim];
	b = new double[_lmaxdim];
	x = new double[_lmaxdim];

	for (int i = 1; i <= free - 1; i++) {
		if (f[(2 - 1) * maxfunc + (i - 1)] > 0) {

			// Get the maximum side length of the hyper rectangle and then set the
			// new side length to this lengths times the growth factor.
			int help = dirGetMaxDeep(i, length, maxfunc, n);
			double sidelen = thirds[help] * 2.;

			// Set the Center and the upper and lower bounds of the rectangles.
			for (int j = 1; j <= n; j++) {
				sidelen = thirds[length[(j - 1) * maxfunc + (i - 1)]];
				a[j - 1] = c[(j - 1) * maxfunc + (i - 1)] - sidelen;
				b[j - 1] = c[(j - 1) * maxfunc + (i - 1)] + sidelen;
			}

			// The function value is reset to 'Inf', since it may have been changed
			// in an earlier iteration and now the feasible point which was close
			// is not close anymore
			f[i - 1] = 1.0e6;
			f[(2 - 1) * maxfunc + (i - 1)] = 2.;

			// Check if any feasible point is near this infeasible point.
			for (int k = 1; k <= free - 1; k++) {

				// If the point k is feasible, check if it is near.
				if (f[(2 - 1) * maxfunc + (k - 1)] == 0) {

					// Copy the coordinates of the point k into x.
					for (int l = 1; l <= n; l++) {
						x[l - 1] = c[(l - 1) * maxfunc + (k - 1)];
					}

					// Check if the point k is near the infeasible point, if so, replace the
					// value at
					if (isInBox(x, a, b, n, _lmaxdim) == 1) {
						f[i - 1] = std::min(f[i - 1], f[k - 1]);
						f[(2 - 1) * maxfunc + (i - 1)] = 1.;
					}
				}
			}
			if (f[(2 - 1) * maxfunc + (i - 1)] == 1.) {
				f[i - 1] = f[i - 1] + 1e-6 * std::fabs(f[i - 1]);
				for (int l = 1; l <= n; l++) {
					x[l - 1] = c[(l - 1) * maxfunc + (i - 1)] * c1[l - 1]
							+ c[(l - 1) * maxfunc + (i - 1)] * c2[l - 1];
				}
				dirResortList(i, anchor, f, point, length, n, maxfunc);
			} else {

				// Replaced fixed value for infeasible points with maximum value found,
				// increased by 1.
				if (!(fmax == f[i - 1])) {
					f[i - 1] = std::max(fmax + 1., f[i - 1]);
				}
			}
		}
	}

	// deallocate
	delete[] a;
	delete[] b;
	delete[] x;
}

int Directl::dirGetMaxDeep(int pos, int *length, int maxfunc, int n) {
	int help = length[pos - 1];
	for (int i = 2; i <= n; i++) {
		help = std::min(help, length[(i - 1) * maxfunc + (pos - 1)]);
	}
	return help;
}

void Directl::dirResortList(int replace, int *anchor, double *f, int *point,
		int *length, int n, int maxfunc) {

	// Get the length of the hyper rectangle with infeasible mid point and
	// Index of the corresponding list.
	int l = dirGetLevel(replace, length, maxfunc, n);
	int start = anchor[l + 1];

	// If the hyper rectangle with infeasibel midpoint is already the start
	// of the list, give out message, nothing to do.
	if (replace == start) {
	} else {

		// Take the hyper rectangle with infeasible midpoint out of the list.
		int pos = start;
		for (int i = 1; i <= maxfunc; i++) {
			if (point[pos - 1] == replace) {
				point[pos - 1] = point[replace - 1];
				goto l20;
			} else {
				pos = point[pos - 1];
			}
			if (pos == 0) {
				goto l20;
			}
		}

		// If the anchor of the list has a higher value than the value of a
		// nearby point, put the infeasible point at the beginning of the list.
		l20: if (f[start - 1] > f[replace - 1]) {
			anchor[l + 1] = replace;
			point[replace - 1] = start;
		} else {

			// Insert the point into the list according to its (replaced) function
			// value.
			pos = start;
			for (int i = 1; i <= maxfunc; i++) {

				// The point has to be added at the end of the list.
				if (point[pos - 1] == 0) {
					point[replace - 1] = point[pos - 1];
					point[pos - 1] = replace;
					return;
				} else {
					if (f[point[pos - 1] - 1] > f[replace - 1]) {
						point[replace - 1] = point[pos - 1];
						point[pos - 1] = replace;
						return;
					}
					pos = point[pos - 1];
				}
			}
		}
	}
}

void Directl::dirInsertList(int &nnew, int *anchor, int *point, double *f,
		int maxi, int *length, int maxfunc, int maxdeep, int n, int samp) {
	for (int j = 1; j <= maxi; j++) {
		int pos1 = nnew;
		int pos2 = point[pos1 - 1];
		nnew = point[pos2 - 1];
		int deep = dirGetLevel(pos1, length, maxfunc, n);
		if (anchor[deep + 1] == 0) {
			if (f[pos2 - 1] < f[pos1 - 1]) {
				anchor[deep + 1] = pos2;
				point[pos2 - 1] = pos1;
				point[pos1 - 1] = 0;
			} else {
				anchor[deep + 1] = pos1;
				point[pos2 - 1] = 0;
			}
		} else {
			int pos = anchor[deep + 1];
			if (f[pos2 - 1] < f[pos1 - 1]) {
				if (f[pos2 - 1] < f[pos1 - 1]) {
					anchor[deep + 1] = pos2;
					if (f[pos1 - 1] < f[pos - 1]) {
						point[pos2 - 1] = pos1;
						point[pos1 - 1] = pos;
					} else {
						point[pos2 - 1] = pos;
						dirInsert(pos, pos1, point, f, maxfunc);
					}
				} else {
					dirInsert(pos, pos2, point, f, maxfunc);
					dirInsert(pos, pos1, point, f, maxfunc);
				}
			} else {
				if (f[pos1 - 1] < f[pos - 1]) {
					anchor[deep + 1] = pos1;
					if (f[pos - 1] < f[pos2 - 1]) {
						point[pos1 - 1] = pos;
						dirInsert(pos, pos2, point, f, maxfunc);
					} else {
						point[pos1 - 1] = pos2;
						point[pos2 - 1] = pos;
					}
				} else {
					dirInsert(pos, pos1, point, f, maxfunc);
					dirInsert(pos, pos2, point, f, maxfunc);
				}
			}
		}
	}
	int deep = dirGetLevel(samp, length, maxfunc, n);
	int pos = anchor[deep + 1];
	if (f[samp - 1] < f[pos - 1]) {
		anchor[deep + 1] = samp;
		point[samp - 1] = pos;
	} else {
		dirInsert(pos, samp, point, f, maxfunc);
	}
}

int Directl::dirGetLevel(int pos, int *length, int maxfunc, int n) {
	if (_jones == 0) {
		int help = length[pos - 1];
		int k = help;
		int p = 1;
		for (int i = 2; i <= n; i++) {
			if (length[(i - 1) * maxfunc + (pos - 1)] < k) {
				k = length[(i - 1) * maxfunc + (pos - 1)];
			}
			if (length[(i - 1) * maxfunc + (pos - 1)] == help) {
				p++;
			}
		}
		if (k == help) {
			return k * n + n - p;
		} else {
			return k * n + p;
		}
	} else {
		int help = length[pos - 1];
		for (int i = 2; i <= n; i++) {
			if (length[(i - 1) * maxfunc + (pos - 1)] < help) {
				help = length[(i - 1) * maxfunc + (pos - 1)];
			}
		}
		return help;
	}
}

void Directl::dirDivide(int nnew, int clen, int *length, int *point,
		int *arrayi, int sample, int *list2, double *w, int maxi, double *f,
		int maxfunc, int maxdeep, int n) {
	int start = 0;
	int pos = nnew;
	int k;
	for (int i = 1; i <= maxi; i++) {
		int j = arrayi[i - 1];
		w[j - 1] = f[pos - 1];
		k = pos;
		pos = point[pos - 1];
		w[j - 1] = std::min(f[pos - 1], w[j - 1]);
		pos = point[pos - 1];
		dirInsertList2(start, j, k, list2, w, maxi, n);
	}
	if (pos > 0) {
		// error
		return;
	}
	for (int j = 1; j <= maxi; j++) {
		dirSearchMin(start, list2, pos, k, n);
		int pos2 = start;
		length[(k - 1) * maxfunc + (sample - 1)] = clen + 1;
		for (int i = 1; i <= maxi - j + 1; i++) {
			length[(k - 1) * maxfunc + (pos - 1)] = clen + 1;
			pos = point[pos - 1];
			length[(k - 1) * maxfunc + (pos - 1)] = clen + 1;
			if (pos2 > 0) {
				pos = list2[(2 - 1) * n + (pos2 - 1)];
				pos2 = list2[pos2 - 1];
			}
		}
	}
}

void Directl::dirInsertList2(int &start, int j, int k, int *list2, double *w,
		int maxi, int n) {
	int pos = start;
	if (start == 0) {
		list2[j - 1] = 0;
		start = j;
		list2[(2 - 1) * n + (j - 1)] = k;
		return;
	}
	if (w[start - 1] > w[j - 1]) {
		list2[j - 1] = start;
		start = j;
	} else {
		for (int i = 1; i <= maxi; i++) {
			if (list2[pos - 1] == 0) {
				list2[j - 1] = 0;
				list2[pos - 1] = j;
				list2[(2 - 1) * n + (j - 1)] = k;
				return;
			} else {
				if (w[j - 1] < w[list2[pos - 1] - 1]) {
					list2[j - 1] = list2[pos - 1];
					list2[pos - 1] = j;
					list2[(2 - 1) * n + (j - 1)] = k;
					return;
				}
			}
			pos = list2[pos - 1];
		}
	}
	list2[(2 - 1) * n + (j - 1)] = k;
}

void Directl::dirSearchMin(int &start, int *list2, int &pos, int &k, int n) {
	k = start;
	pos = list2[(2 - 1) * n + (start - 1)];
	start = list2[start - 1];
}

void Directl::dirSamplePoints(double *c, int *arrayi, double delta, int sample,
		int &start, int *length, int &free, int maxi, int *point, int n,
		int maxfunc, int &oops) {
	oops = 0;
	int pos = free;
	start = free;
	for (int k = 1; k <= maxi + maxi; k++) {
		for (int j = 1; j <= n; j++) {
			length[(j - 1) * maxfunc + (free - 1)] = length[(j - 1) * maxfunc
					+ (sample - 1)];
			c[(j - 1) * maxfunc + (free - 1)] = c[(j - 1) * maxfunc
					+ (sample - 1)];
		}
		pos = free;
		free = point[free - 1];
		if (free == 0) {
			oops = 1;
			return;
		}
	}
	point[pos - 1] = 0;
	pos = start;
	for (int j = 1; j <= maxi; j++) {
		c[(arrayi[j - 1] - 1) * maxfunc + (pos - 1)] = c[(arrayi[j - 1] - 1)
				* maxfunc + (sample - 1)] + delta;
		pos = point[pos - 1];
		c[(arrayi[j - 1] - 1) * maxfunc + (pos - 1)] = c[(arrayi[j - 1] - 1)
				* maxfunc + (sample - 1)] - delta;
		pos = point[pos - 1];
	}
	if (pos > 0) {
		throw std::invalid_argument("Error [DIRECT]: pos > 0.");
	}
}

void Directl::dirGetI(int *length, int pos, int *arrayi, int &maxi, int n,
		int maxfunc) {
	int j = 1;
	int help = length[pos - 1];
	for (int i = 2; i <= n; i++) {
		if (length[(i - 1) * maxfunc + (pos - 1)] < help) {
			help = length[(i - 1) * maxfunc + (pos - 1)];
		}
	}
	for (int i = 1; i <= n; i++) {
		if (length[(i - 1) * maxfunc + (pos - 1)] == help) {
			arrayi[j - 1] = i;
			j++;
		}
	}
	maxi = j - 1;
}

void Directl::dirInitList(int *anchor, int &free, int *point, double *f,
		int maxfunc, int maxdeep) {
	for (int i = -1; i <= maxdeep; i++) {
		anchor[i + 1] = 0;
	}
	for (int i = 1; i <= maxfunc; i++) {
		f[i - 1] = 0.;
		f[i - 1 + maxfunc] = 0;
		point[i - 1] = i + 1;
	}
	point[maxfunc - 1] = 0;
	free = 1;
}

void Directl::dirInsert3(int &pos1, int &pos2, int &pos3, int deep, int *anchor,
		int *point, int free, double *f, double &fmin, int &minpos,
		int maxfunc) {
	dirSort3(pos1, pos2, pos3, f);
	if (anchor[deep + 1] == 0) {
		anchor[deep + 1] = pos1;
		point[pos1 - 1] = pos2;
		point[pos2 - 1] = pos3;
		point[pos3 - 1] = 0;
	} else {
		int pos = anchor[deep + 1];
		if (f[pos1 - 1] < f[pos - 1]) {
			anchor[deep + 1] = pos1;
			point[pos1 - 1] = pos;
		} else {
			dirInsert(pos, pos1, point, f, maxfunc);
		}
		dirInsert(pos, pos2, point, f, maxfunc);
		dirInsert(pos, pos3, point, f, maxfunc);
	}
	if (f[pos1 - 1] < fmin && f[pos1 - 1 + maxfunc] == 0) {
		fmin = f[pos1 - 1];
		minpos = pos1;
	}
}

void Directl::dirInsert(int &start, int ins, int *point, double *f,
		int maxfunc) {
	for (int i = 1; i <= maxfunc; i++) {
		if (point[start - 1] == 0) {
			point[start - 1] = ins;
			point[ins - 1] = 0;
			return;
		} else {
			if (f[ins - 1] < f[point[start - 1] - 1]) {
				const int help = point[start - 1];
				point[start - 1] = ins;
				point[ins - 1] = help;
				return;
			}
		}
		start = point[start - 1];
	}
}

void Directl::dirSort3(int &pos1, int &pos2, int &pos3, double *f) {
	if (f[pos1 - 1] < f[pos2 - 1]) {
		if (f[pos1 - 1] < f[pos3 - 1]) {
			if (f[pos3 - 1] < f[pos2 - 1]) {
				int help = pos2;
				pos2 = pos3;
				pos3 = help;
			}
		} else {
			int help = pos1;
			pos1 = pos3;
			pos3 = pos2;
			pos2 = help;
		}
	} else {
		if (f[pos2 - 1] < f[pos3 - 1]) {
			if (f[pos3 - 1] < f[pos1 - 1]) {
				int help = pos1;
				pos1 = pos2;
				pos2 = pos3;
				pos3 = help;
			} else {
				int help = pos1;
				pos1 = pos2;
				pos2 = help;
			}
		} else {
			int help = pos1;
			pos1 = pos3;
			pos3 = help;
		}
	}
}

int Directl::dirPreprc(double *u, double *l, int n, double *xs1, double *xs2) {
	int oops = 0;

	// Check if the hyper-box is well-defined.
	for (int i = 1; i <= n; i++) {
		if (u[i - 1] <= l[i - 1]) {
			oops = 1;
			return oops;
		}
	}

	// Scale the initial iterate so that it is in the unit cube.
	for (int i = 1; i <= n; i++) {
		const double help = u[i - 1] - l[i - 1];
		xs2[i - 1] = l[i - 1] / help;
		xs1[i - 1] = help;
	}

	return oops;
}

double Directl::dirInfcn(const multivariate &f, double *x, double *c1,
		double *c2, int n) {

	// Unscale the variable x.
	for (int i = 1; i <= n; i++) {
		x[i - 1] = (x[i - 1] + c2[i - 1]) * c1[i - 1];
	}

	// Call the function-evaluation subroutine.
	double res = f(x);

	// Rescale the variable x.
	for (int i = 1; i <= n; i++) {
		x[i - 1] = x[i - 1] / c1[i - 1] - c2[i - 1];
	}

	return res;
}

int Directl::isInBox(double *x, double *a, double *b, int n, int lmd) {
	int res = 1;
	for (int i = 1; i <= n; i++) {
		if (a[i - 1] > x[i - 1] || b[i - 1] < x[i - 1]) {
			res = 0;
			break;
		}
	}
	return res;
}
