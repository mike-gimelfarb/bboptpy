/*
 Original source code in C++ from https://github.com/elsid/bobyqa-cpp

 The MIT License (MIT)

 Copyright (c) 2015

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <algorithm>

#include "bobyqa.h"

Bobyqa::Bobyqa(int mfev, int np, double rho, double tol) {
	_np = np;
	_mfev = mfev;
	_rho = rho;
	_tol = tol;
}

void Bobyqa::init(const multivariate_problem &f, const double *guess) {
	// nothing to do here
}

void Bobyqa::iterate() {
	// nothing to do here
}

multivariate_solution Bobyqa::solution(){
	return {{}, 0, false};
}

multivariate_solution Bobyqa::optimize(const multivariate_problem &f,
		const double *guess) {
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [BOBYQA]: problem constraints will be ignored."
				<< std::endl;
	}

	// initialize point
	std::vector<double> x(guess, guess + f._n);

	// adjust population size
	int npt = _np;
	if (npt < f._n + 2) {
		npt = f._n + 2;
		std::cerr << "Warning [BOBYQA]: population size too small - adjusted."
				<< std::endl;
	} else if (npt > (f._n + 2) * (f._n + 1) / 2) {
		npt = (f._n + 2) * (f._n + 1) / 2;
		std::cerr << "Warning [BOBYQA]: population size too large - adjusted."
				<< std::endl;
	}

	// adjust rhobeg
	bool warned = false;
	for (int i = 0; i < f._n; i++) {
		if (_rho <= 0. || f._upper[i] - f._lower[i] < _rho + _rho) {
			_rho = (f._upper[i] - f._lower[i]) / 2.;
			if (!warned) {
				std::cerr
						<< "Warning [BOBYQA]: rho is inadequate - adjusted."
						<< std::endl;
				warned = true;
			}
		}
	}

	// create work array
	std::vector<double> w((npt + 5) * (npt + f._n) + 3 * f._n * (f._n + 5) / 2);

	// main call
	long fev = 0;
	bobyqa(f._f, static_cast<long>(f._n), static_cast<long>(npt), &x[0],
			&f._lower[0], &f._upper[0], _rho, _tol, static_cast<long>(_mfev),
			&w[0], fev);
	return {x, static_cast<int>(fev), false};
}

void Bobyqa::altmov(const long n, const long npt, const double *xpt,
		const double *const xopt, const double *bmat, const double *zmat,
		const long ndim, const double *const sl, const double *const su,
		const long kopt, const long knew, const double adelt,
		double *const xnew, double *const xalt, double &alpha, double &cauchy,
		double *const glag, double *const hcol, double *const w) {

	/* Local variables */
	double gw, diff;
	long ilbd, isbd;
	double slbd;
	long iubd;
	double vlag, subd, temp;
	long ksav = 0;
	double step = 0, curv = 0;
	long iflag;
	double scale = 0, csave = 0, tempa = 0, tempb = 0, tempd = 0, sumin = 0,
			ggfree = 0;
	long ibdsav = 0;
	double dderiv = 0, bigstp = 0, predsq = 0, presav = 0, distsq = 0, stpsav =
			0, wfixsq = 0, wsqsav = 0;

	/*     The arguments N, NPT, XPT, XOPT, BMAT, ZMAT, NDIM, SL and SU all have */
	/*       the same meanings as the corresponding arguments of BOBYQB. */
	/*     KOPT is the index of the optimal interpolation point. */
	/*     KNEW is the index of the interpolation point that is going to be moved. */
	/*     ADELT is the current trust region bound. */
	/*     XNEW will be set to a suitable new position for the interpolation point */
	/*       XPT(KNEW,.). Specifically, it satisfies the SL, SU and trust region */
	/*       bounds and it should provide a large denominator in the next call of */
	/*       UPDATE. The step XNEW-XOPT from XOPT is restricted to moves along the */
	/*       straight lines through XOPT and another interpolation point. */
	/*     XALT also provides a large value of the modulus of the KNEW-th Lagrange */
	/*       function subject to the constraints that have been mentioned, its main */
	/*       difference from XNEW being that XALT-XOPT is a constrained version of */
	/*       the Cauchy step within the trust region. An exception is that XALT is */
	/*       not calculated if all components of GLAG (see below) are zero. */
	/*     ALPHA will be set to the KNEW-th diagonal element of the H matrix. */
	/*     CAUCHY will be set to the square of the KNEW-th Lagrange function at */
	/*       the step XALT-XOPT from XOPT for the vector XALT that is returned, */
	/*       except that CAUCHY is set to zero if XALT is not calculated. */
	/*     GLAG is a working space vector of length N for the gradient of the */
	/*       KNEW-th Lagrange function at XOPT. */
	/*     HCOL is a working space vector of length NPT for the second derivative */
	/*       coefficients of the KNEW-th Lagrange function. */
	/*     W is a working space vector of length 2N that is going to hold the */
	/*       constrained Cauchy step from XOPT of the Lagrange function, followed */
	/*       by the downhill version of XALT when the uphill step is calculated. */

	/*     Set the first NPT components of W to the leading elements of the */
	/*     KNEW-th column of the H matrix. */

	/* Parameter adjustments */
	const long zmat_dim1 = npt;
	const long zmat_offset = 1 + zmat_dim1;
	zmat -= zmat_offset;
	const long xpt_dim1 = npt;
	const long xpt_offset = 1 + xpt_dim1;
	xpt -= xpt_offset;
	const long bmat_dim1 = ndim;
	const long bmat_offset = 1 + bmat_dim1;
	bmat -= bmat_offset;

	/* Function Body */
	for (long k = 1; k <= npt; ++k) {
		hcol[k] = 0.;
	}
	const long j_n = npt - n - 1;
	for (long j = 1; j <= j_n; ++j) {
		temp = zmat[knew + j * zmat_dim1];
		for (long k = 1; k <= npt; ++k) {
			hcol[k] += temp * zmat[k + j * zmat_dim1];
		}
	}
	alpha = hcol[knew];
	const double ha = 0.5 * alpha;

	/*     Calculate the gradient of the KNEW-th Lagrange function at XOPT. */
	for (long i = 1; i <= n; ++i) {
		glag[i] = bmat[knew + i * bmat_dim1];
	}
	for (long k = 1; k <= npt; ++k) {
		temp = 0.;
		for (long j = 1; j <= n; ++j) {
			temp += xpt[k + j * xpt_dim1] * xopt[j];
		}
		temp = hcol[k] * temp;
		for (long i = 1; i <= n; ++i) {
			glag[i] += temp * xpt[k + i * xpt_dim1];
		}
	}

	/*     Search for a large denominator along the straight lines through XOPT */
	/*     and another interpolation point. SLBD and SUBD will be lower and upper */
	/*     bounds on the step along each of these lines in turn. PREDSQ will be */
	/*     set to the square of the predicted denominator for each line. PRESAV */
	/*     will be set to the largest admissible value of PREDSQ that occurs. */
	presav = 0.;
	for (long k = 1; k <= npt; ++k) {
		if (k == kopt) {
			goto L80;
		}
		dderiv = 0.;
		distsq = 0.;
		for (long i = 1; i <= n; ++i) {
			temp = xpt[k + i * xpt_dim1] - xopt[i];
			dderiv += glag[i] * temp;
			distsq += temp * temp;
		}
		subd = adelt / std::sqrt(distsq);
		slbd = -subd;
		ilbd = 0;
		iubd = 0;
		sumin = std::min(1., subd);

		/*     Revise SLBD and SUBD if necessary because of the bounds in SL and SU. */
		for (long i = 1; i <= n; ++i) {
			temp = xpt[k + i * xpt_dim1] - xopt[i];
			if (temp > 0.) {
				if (slbd * temp < sl[i] - xopt[i]) {
					slbd = (sl[i] - xopt[i]) / temp;
					ilbd = -i;
				}
				if (subd * temp > su[i] - xopt[i]) {
					subd = std::max(sumin, (su[i] - xopt[i]) / temp);
					iubd = i;
				}
			} else if (temp < 0.) {
				if (slbd * temp > su[i] - xopt[i]) {
					slbd = (su[i] - xopt[i]) / temp;
					ilbd = i;
				}
				if (subd * temp < sl[i] - xopt[i]) {
					subd = std::max(sumin, (sl[i] - xopt[i]) / temp);
					iubd = -i;
				}
			}
		}

		/*     Seek a large modulus of the KNEW-th Lagrange function when the index */
		/*     of the other interpolation point on the line through XOPT is KNEW. */
		if (k == knew) {
			diff = dderiv - 1.;
			step = slbd;
			vlag = slbd * (dderiv - slbd * diff);
			isbd = ilbd;
			temp = subd * (dderiv - subd * diff);
			if (std::abs(temp) > std::abs(vlag)) {
				step = subd;
				vlag = temp;
				isbd = iubd;
			}
			tempd = 0.5 * dderiv;
			tempa = tempd - diff * slbd;
			tempb = tempd - diff * subd;
			if (tempa * tempb < 0.) {
				temp = tempd * tempd / diff;
				if (std::abs(temp) > std::abs(vlag)) {
					step = tempd / diff;
					vlag = temp;
					isbd = 0;
				}
			}

			/*     Search along each of the other lines through XOPT and another point. */
		} else {
			step = slbd;
			vlag = slbd * (1. - slbd);
			isbd = ilbd;
			temp = subd * (1. - subd);
			if (std::abs(temp) > std::abs(vlag)) {
				step = subd;
				vlag = temp;
				isbd = iubd;
			}
			if (subd > 0.5) {
				if (std::abs(vlag) < 0.25) {
					step = 0.5;
					vlag = 0.25;
					isbd = 0;
				}
			}
			vlag *= dderiv;
		}

		/*     Calculate PREDSQ for the current line search and maintain PRESAV. */
		temp = step * (1. - step) * distsq;
		predsq = vlag * vlag * (vlag * vlag + ha * temp * temp);
		if (predsq > presav) {
			presav = predsq;
			ksav = k;
			stpsav = step;
			ibdsav = isbd;
		}
		L80: ;
	}

	/*     Construct XNEW in a way that satisfies the bound constraints exactly. */
	for (long i = 1; i <= n; ++i) {
		temp = xopt[i] + stpsav * (xpt[ksav + i * xpt_dim1] - xopt[i]);
		xnew[i] = std::max(sl[i], std::min(su[i], temp));
	}
	if (ibdsav < 0) {
		xnew[-ibdsav] = sl[-ibdsav];
	}
	if (ibdsav > 0) {
		xnew[ibdsav] = su[ibdsav];
	}

	/*     Prepare for the iterative method that assembles the constrained Cauchy */
	/*     step in W. The sum of squares of the fixed components of W is formed in */
	/*     WFIXSQ, and the free components of W are set to BIGSTP. */
	bigstp = adelt + adelt;
	iflag = 0;

	L100: wfixsq = 0.;
	ggfree = 0.;
	for (long i = 1; i <= n; ++i) {
		w[i] = 0.;
		tempa = std::min(xopt[i] - sl[i], glag[i]);
		tempb = std::max(xopt[i] - su[i], glag[i]);
		if (tempa > 0. || tempb < 0.) {
			w[i] = bigstp;
			ggfree += glag[i] * glag[i];
		}
	}
	if (ggfree == 0.) {
		cauchy = 0.;
		goto L200;
	}

	/*     Investigate whether more components of W can be fixed. */
	L120: temp = adelt * adelt - wfixsq;
	if (temp > 0.) {
		wsqsav = wfixsq;
		step = std::sqrt(temp / ggfree);
		ggfree = 0.;
		for (long i = 1; i <= n; ++i) {
			if (w[i] == bigstp) {
				temp = xopt[i] - step * glag[i];
				if (temp <= sl[i]) {
					w[i] = sl[i] - xopt[i];
					wfixsq += w[i] * w[i];
				} else if (temp >= su[i]) {
					w[i] = su[i] - xopt[i];
					wfixsq += w[i] * w[i];
				} else {
					ggfree += glag[i] * glag[i];
				}
			}
		}
		if (wfixsq > wsqsav && ggfree > 0.) {
			goto L120;
		}
	}

	/*     Set the remaining free components of W and all components of XALT, */
	/*     except that W may be scaled later. */
	gw = 0.;
	for (long i = 1; i <= n; ++i) {
		if (w[i] == bigstp) {
			w[i] = -step * glag[i];
			xalt[i] = std::max(sl[i], std::min(su[i], xopt[i] + w[i]));
		} else if (w[i] == 0.) {
			xalt[i] = xopt[i];
		} else if (glag[i] > 0.) {
			xalt[i] = sl[i];
		} else {
			xalt[i] = su[i];
		}
		gw += glag[i] * w[i];
	}

	/*     Set CURV to the curvature of the KNEW-th Lagrange function along W. */
	/*     Scale W by a factor less than one if that can reduce the modulus of */
	/*     the Lagrange function at XOPT+W. Set CAUCHY to the final value of */
	/*     the square of this function. */
	curv = 0.;
	for (long k = 1; k <= npt; ++k) {
		temp = 0.;
		for (long j = 1; j <= n; ++j) {
			temp += xpt[k + j * xpt_dim1] * w[j];
		}
		curv += hcol[k] * temp * temp;
	}
	if (iflag == 1) {
		curv = -curv;
	}
	if (curv > -gw && curv < -(1. + std::sqrt(2.)) * gw) {
		scale = -gw / curv;
		for (long i = 1; i <= n; ++i) {
			temp = xopt[i] + scale * w[i];
			xalt[i] = std::max(sl[i], std::min(su[i], temp));
		}
		cauchy = std::pow(0.5 * gw * scale, 2);
	} else {
		cauchy = std::pow(gw + 0.5 * curv, 2);
	}

	/*     If IFLAG is zero, then XALT is calculated as before after reversing */
	/*     the sign of GLAG. Thus two XALT vectors become available. The one that */
	/*     is chosen is the one that gives the larger value of CAUCHY. */
	if (iflag == 0) {
		for (long i = 1; i <= n; ++i) {
			glag[i] = -glag[i];
			w[n + i] = xalt[i];
		}
		csave = cauchy;
		iflag = 1;
		goto L100;
	}
	if (csave > cauchy) {
		for (long i = 1; i <= n; ++i) {
			xalt[i] = w[n + i];
		}
		cauchy = csave;
	}
	L200: ;
}

void Bobyqa::prelim(const multivariate &function, const long n, const long npt,
		double *const x, const double *const xl, const double *const xu,
		const double rhobeg, const long maxfun, double *const xbase,
		double *xpt, double *const fval, double *const gopt, double *const hq,
		double *const pq, double *bmat, double *zmat, const long ndim,
		const double *const sl, const double *const su, long &nf, long &kopt,
		long &fev) {

	/* Local variables */
	long ih, nfm;
	long nfx = 0, ipt = 0, jpt = 0;
	double fbeg = 0, diff = 0, temp = 0, stepa = 0, stepb = 0;
	long itemp;

	/*     The arguments N, NPT, X, XL, XU, RHOBEG, IPRINT and MAXFUN are the */
	/*       same as the corresponding arguments in SUBROUTINE BOBYQA. */
	/*     The arguments XBASE, XPT, FVAL, HQ, PQ, BMAT, ZMAT, NDIM, SL and SU */
	/*       are the same as the corresponding arguments in BOBYQB, the elements */
	/*       of SL and SU being set in BOBYQA. */
	/*     GOPT is usually the gradient of the quadratic model at XOPT+XBASE, but */
	/*       it is set by PRELIM to the gradient of the quadratic model at XBASE. */
	/*       If XOPT is nonzero, BOBYQB will change it to its usual value later. */
	/*     NF is maintaned as the number of calls of CALFUN so far. */
	/*     KOPT will be such that the least calculated value of F so far is at */
	/*       the point XPT(KOPT,.)+XBASE in the space of the variables. */

	/*     SUBROUTINE PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ, */
	/*     BMAT and ZMAT for the first iteration, and it maintains the values of */
	/*     NF and KOPT. The vector X is also changed by PRELIM. */

	/*     Set some constants. */

	/* Parameter adjustments */
	const long zmat_dim1 = npt;
	const long zmat_offset = 1 + zmat_dim1;
	zmat -= zmat_offset;
	const long xpt_dim1 = npt;
	const long xpt_offset = 1 + xpt_dim1;
	xpt -= xpt_offset;
	const long bmat_dim1 = ndim;
	const long bmat_offset = 1 + bmat_dim1;
	bmat -= bmat_offset;

	/* Function Body */
	const double rhosq = rhobeg * rhobeg;
	const double recip = 1. / rhosq;
	const long np = n + 1;

	/*     Set XBASE to the initial vector of variables, and set the initial */
	/*     elements of XPT, BMAT, HQ, PQ and ZMAT to zero. */
	for (long j = 1; j <= n; ++j) {
		xbase[j] = x[j];
		for (long k = 1; k <= npt; ++k) {
			xpt[k + j * xpt_dim1] = 0.;
		}
		for (long i = 1; i <= ndim; ++i) {
			bmat[i + j * bmat_dim1] = 0.;
		}
	}
	const long ih_n = n * np / 2;
	for (long ih = 1; ih <= ih_n; ++ih) {
		hq[ih] = 0.;
	}
	for (long k = 1; k <= npt; ++k) {
		pq[k] = 0.;
		const long j_n = npt - np;
		for (long j = 1; j <= j_n; ++j) {
			zmat[k + j * zmat_dim1] = 0.;
		}
	}

	/*     Begin the initialization procedure. NF becomes one more than the number */
	/*     of function values so far. The coordinates of the displacement of the */
	/*     next initial interpolation point from XBASE are set in XPT(NF+1,.). */
	nf = 0;

	L50: nfm = nf;
	nfx = nf - n;
	++nf;
	if (nfm <= n << 1) {
		if (nfm >= 1 && nfm <= n) {
			stepa = rhobeg;
			if (su[nfm] == 0.) {
				stepa = -stepa;
			}
			xpt[nf + nfm * xpt_dim1] = stepa;
		} else if (nfm > n) {
			stepa = xpt[nf - n + nfx * xpt_dim1];
			stepb = -rhobeg;
			if (sl[nfx] == 0.) {
				stepb = std::min(2. * rhobeg, su[nfx]);
			}
			if (su[nfx] == 0.) {
				stepb = std::max(-2. * rhobeg, sl[nfx]);
			}
			xpt[nf + nfx * xpt_dim1] = stepb;
		}
	} else {
		itemp = (nfm - np) / n;
		jpt = nfm - itemp * n - n;
		ipt = jpt + itemp;
		if (ipt > n) {
			itemp = jpt;
			jpt = ipt - n;
			ipt = itemp;
		}
		xpt[nf + ipt * xpt_dim1] = xpt[ipt + 1 + ipt * xpt_dim1];
		xpt[nf + jpt * xpt_dim1] = xpt[jpt + 1 + jpt * xpt_dim1];
	}

	/*     Calculate the next value of F. The least function value so far and */
	/*     its index are required. */
	for (long j = 1; j <= n; ++j) {
		x[j] = std::min(std::max(xl[j], xbase[j] + xpt[nf + j * xpt_dim1]),
				xu[j]);
		if (xpt[nf + j * xpt_dim1] == sl[j]) {
			x[j] = xl[j];
		}
		if (xpt[nf + j * xpt_dim1] == su[j]) {
			x[j] = xu[j];
		}
	}
	const double f = function(x + 1);
	fev++;
	fval[nf] = f;
	if (nf == 1) {
		fbeg = f;
		kopt = 1;
	} else if (f < fval[kopt]) {
		kopt = nf;
	}

	/*     Set the nonzero initial elements of BMAT and the quadratic model in the */
	/*     cases when NF is at most 2*N+1. If NF exceeds N+1, then the positions */
	/*     of the NF-th and (NF-N)-th interpolation points may be switched, in */
	/*     order that the function value at the first of them contributes to the */
	/*     off-diagonal second derivative terms of the initial quadratic model. */
	if (nf <= (n << 1) + 1) {
		if (nf >= 2 && nf <= n + 1) {
			gopt[nfm] = (f - fbeg) / stepa;
			if (npt < nf + n) {
				bmat[nfm * bmat_dim1 + 1] = -1. / stepa;
				bmat[nf + nfm * bmat_dim1] = 1. / stepa;
				bmat[npt + nfm + nfm * bmat_dim1] = -0.5 * rhosq;
			}
		} else if (nf >= n + 2) {
			ih = nfx * (nfx + 1) / 2;
			temp = (f - fbeg) / stepb;
			diff = stepb - stepa;
			hq[ih] = 2. * (temp - gopt[nfx]) / diff;
			gopt[nfx] = (gopt[nfx] * stepb - temp * stepa) / diff;
			if (stepa * stepb < 0.) {
				if (f < fval[nf - n]) {
					fval[nf] = fval[nf - n];
					fval[nf - n] = f;
					if (kopt == nf) {
						kopt = nf - n;
					}
					xpt[nf - n + nfx * xpt_dim1] = stepb;
					xpt[nf + nfx * xpt_dim1] = stepa;
				}
			}
			bmat[nfx * bmat_dim1 + 1] = -(stepa + stepb) / (stepa * stepb);
			bmat[nf + nfx * bmat_dim1] = -0.5 / xpt[nf - n + nfx * xpt_dim1];
			bmat[nf - n + nfx * bmat_dim1] = -bmat[nfx * bmat_dim1 + 1]
					- bmat[nf + nfx * bmat_dim1];
			zmat[nfx * zmat_dim1 + 1] = std::sqrt(2.) / (stepa * stepb);
			zmat[nf + nfx * zmat_dim1] = std::sqrt(0.5) / rhosq;
			zmat[nf - n + nfx * zmat_dim1] = -zmat[nfx * zmat_dim1 + 1]
					- zmat[nf + nfx * zmat_dim1];
		}

		/*     Set the off-diagonal second derivatives of the Lagrange functions and */
		/*     the initial quadratic model. */
	} else {
		ih = ipt * (ipt - 1) / 2 + jpt;
		zmat[nfx * zmat_dim1 + 1] = recip;
		zmat[nf + nfx * zmat_dim1] = recip;
		zmat[ipt + 1 + nfx * zmat_dim1] = -recip;
		zmat[jpt + 1 + nfx * zmat_dim1] = -recip;
		temp = xpt[nf + ipt * xpt_dim1] * xpt[nf + jpt * xpt_dim1];
		hq[ih] = (fbeg - fval[ipt + 1] - fval[jpt + 1] + f) / temp;
	}
	if (nf < npt && nf < maxfun) {
		goto L50;
	}
}

void Bobyqa::rescue(const multivariate &function, const long n, const long npt,
		const double *const xl, const double *const xu, const long maxfun,
		double *const xbase, double *xpt, double *const fval,
		double *const xopt, double *const gopt, double *const hq,
		double *const pq, double *bmat, double *zmat, const long ndim,
		double *const sl, double *const su, long &nf, const double delta,
		long &kopt, double *const vlag, double *const ptsaux,
		double *const ptsid, double *const w, long &fev) {

	/* Local variables */
	long ih, jp, iq, iw;
	double xp = 0, xq = 0, den = 0;
	long ihp = 0;
	long ihq, jpn;
	double sum = 0, diff = 0, beta = 0;
	long kold;
	double winc;
	long nrem, knew;
	double temp, bsum;
	double hdiag = 0, fbase = 0, denom = 0, vquad = 0, sumpq = 0;
	double dsqmin, distsq, vlmxsq;

	/*     The arguments N, NPT, XL, XU, IPRINT, MAXFUN, XBASE, XPT, FVAL, XOPT, */
	/*       GOPT, HQ, PQ, BMAT, ZMAT, NDIM, SL and SU have the same meanings as */
	/*       the corresponding arguments of BOBYQB on the entry to RESCUE. */
	/*     NF is maintained as the number of calls of CALFUN so far, except that */
	/*       NF is set to -1 if the value of MAXFUN prevents further progress. */
	/*     KOPT is maintained so that FVAL(KOPT) is the least calculated function */
	/*       value. Its correct value must be given on entry. It is updated if a */
	/*       new least function value is found, but the corresponding changes to */
	/*       XOPT and GOPT have to be made later by the calling program. */
	/*     DELTA is the current trust region radius. */
	/*     VLAG is a working space vector that will be used for the values of the */
	/*       provisional Lagrange functions at each of the interpolation points. */
	/*       They are part of a product that requires VLAG to be of length NDIM. */
	/*     PTSAUX is also a working space array. For J=1,2,...,N, PTSAUX(1,J) and */
	/*       PTSAUX(2,J) specify the two positions of provisional interpolation */
	/*       points when a nonzero step is taken along e_J (the J-th coordinate */
	/*       direction) through XBASE+XOPT, as specified below. Usually these */
	/*       steps have length DELTA, but other lengths are chosen if necessary */
	/*       in order to satisfy the given bounds on the variables. */
	/*     PTSID is also a working space array. It has NPT components that denote */
	/*       provisional new positions of the original interpolation points, in */
	/*       case changes are needed to restore the linear independence of the */
	/*       interpolation conditions. The K-th point is a candidate for change */
	/*       if and only if PTSID(K) is nonzero. In this case let p and q be the */
	/*       long parts of PTSID(K) and (PTSID(K)-p) multiplied by N+1. If p */
	/*       and q are both positive, the step from XBASE+XOPT to the new K-th */
	/*       interpolation point is PTSAUX(1,p)*e_p + PTSAUX(1,q)*e_q. Otherwise */
	/*       the step is PTSAUX(1,p)*e_p or PTSAUX(2,q)*e_q in the cases q=0 or */
	/*       p=0, respectively. */
	/*     The first NDIM+NPT elements of the array W are used for working space. */
	/*     The final elements of BMAT and ZMAT are set in a well-conditioned way */
	/*       to the values that are appropriate for the new interpolation points. */
	/*     The elements of GOPT, HQ and PQ are also revised to the values that are */
	/*       appropriate to the final quadratic model. */

	/*     Set some constants. */

	/* Parameter adjustments */
	const long zmat_dim1 = npt;
	const long zmat_offset = 1 + zmat_dim1;
	zmat -= zmat_offset;
	const long xpt_dim1 = npt;
	const long xpt_offset = 1 + xpt_dim1;
	xpt -= xpt_offset;
	const long bmat_dim1 = ndim;
	const long bmat_offset = 1 + bmat_dim1;
	bmat -= bmat_offset;

	/* Function Body */
	const long np = n + 1;
	const double sfrac = 0.5 / double(np);
	const long nptm = npt - np;

	/*     Shift the interpolation points so that XOPT becomes the origin, and set */
	/*     the elements of ZMAT to zero. The value of SUMPQ is required in the */
	/*     updating of HQ below. The squares of the distances from XOPT to the */
	/*     other interpolation points are set at the end of W. Increments of WINC */
	/*     may be added later to these squares to balance the consideration of */
	/*     the choice of point that is going to become current. */
	sumpq = 0.;
	winc = 0.;
	for (long k = 1; k <= npt; ++k) {
		distsq = 0.;
		for (long j = 1; j <= n; ++j) {
			xpt[k + j * xpt_dim1] -= xopt[j];
			distsq += std::pow(xpt[k + j * xpt_dim1], 2);
		}
		sumpq += pq[k];
		w[ndim + k] = distsq;
		winc = std::max(winc, distsq);
		for (long j = 1; j <= nptm; ++j) {
			zmat[k + j * zmat_dim1] = 0.;
		}
	}

	/*     Update HQ so that HQ and PQ define the second derivatives of the model */
	/*     after XBASE has been shifted to the trust region centre. */
	ih = 0;
	for (long j = 1; j <= n; ++j) {
		w[j] = 0.5 * sumpq * xopt[j];
		for (long k = 1; k <= npt; ++k) {
			w[j] += pq[k] * xpt[k + j * xpt_dim1];
		}
		for (long i = 1; i <= j; ++i) {
			++ih;
			hq[ih] = hq[ih] + w[i] * xopt[j] + w[j] * xopt[i];
		}
	}

	/*     Shift XBASE, SL, SU and XOPT. Set the elements of BMAT to zero, and */
	/*     also set the elements of PTSAUX. */
	for (long j = 1; j <= n; ++j) {
		xbase[j] += xopt[j];
		sl[j] -= xopt[j];
		su[j] -= xopt[j];
		xopt[j] = 0.;
		ptsaux[(j << 1) + 1] = std::min(delta, su[j]);
		ptsaux[(j << 1) + 2] = std::max(-delta, sl[j]);
		if (ptsaux[(j << 1) + 1] + ptsaux[(j << 1) + 2] < 0.) {
			temp = ptsaux[(j << 1) + 1];
			ptsaux[(j << 1) + 1] = ptsaux[(j << 1) + 2];
			ptsaux[(j << 1) + 2] = temp;
		}
		if (std::abs(ptsaux[(j << 1) + 2])
				< 0.5 * std::abs(ptsaux[(j << 1) + 1])) {
			ptsaux[(j << 1) + 2] = 0.5 * ptsaux[(j << 1) + 1];
		}
		for (long i = 1; i <= ndim; ++i) {
			bmat[i + j * bmat_dim1] = 0.;
		}
	}
	fbase = fval[kopt];

	/*     Set the identifiers of the artificial interpolation points that are */
	/*     along a coordinate direction from XOPT, and set the corresponding */
	/*     nonzero elements of BMAT and ZMAT. */
	ptsid[1] = sfrac;
	for (long j = 1; j <= n; ++j) {
		jp = j + 1;
		jpn = jp + n;
		ptsid[jp] = double(j + sfrac);
		if (jpn <= npt) {
			ptsid[jpn] = double(j) / double(np) + sfrac;
			temp = 1. / (ptsaux[(j << 1) + 1] - ptsaux[(j << 1) + 2]);
			bmat[jp + j * bmat_dim1] = -temp + 1. / ptsaux[(j << 1) + 1];
			bmat[jpn + j * bmat_dim1] = temp + 1. / ptsaux[(j << 1) + 2];
			bmat[j * bmat_dim1 + 1] = -bmat[jp + j * bmat_dim1]
					- bmat[jpn + j * bmat_dim1];
			zmat[j * zmat_dim1 + 1] = std::sqrt(2.)
					/ std::abs(ptsaux[(j << 1) + 1] * ptsaux[(j << 1) + 2]);
			zmat[jp + j * zmat_dim1] = zmat[j * zmat_dim1 + 1]
					* ptsaux[(j << 1) + 2] * temp;
			zmat[jpn + j * zmat_dim1] = -zmat[j * zmat_dim1 + 1]
					* ptsaux[(j << 1) + 1] * temp;
		} else {
			bmat[j * bmat_dim1 + 1] = -1. / ptsaux[(j << 1) + 1];
			bmat[jp + j * bmat_dim1] = 1. / ptsaux[(j << 1) + 1];
			bmat[j + npt + j * bmat_dim1] = -0.5
					* std::pow(ptsaux[(j << 1) + 1], 2);
		}
	}

	/*     Set any remaining identifiers with their nonzero elements of ZMAT. */
	if (npt >= n + np) {
		for (long k = np << 1; k <= npt; ++k) {
			iw = long((double(k - np) - 0.5) / double(n));
			const long ip = k - np - iw * n;
			iq = ip + iw;
			if (iq > n) {
				iq -= n;
			}
			ptsid[k] = double(ip) + double(iq) / double(np) + sfrac;
			temp = 1. / (ptsaux[(ip << 1) + 1] * ptsaux[(iq << 1) + 1]);
			zmat[(k - np) * zmat_dim1 + 1] = temp;
			zmat[ip + 1 + (k - np) * zmat_dim1] = -temp;
			zmat[iq + 1 + (k - np) * zmat_dim1] = -temp;
			zmat[k + (k - np) * zmat_dim1] = temp;
		}
	}
	nrem = npt;
	kold = 1;
	knew = kopt;

	/*     Reorder the provisional points in the way that exchanges PTSID(KOLD) */
	/*     with PTSID(KNEW). */
	L80: for (long j = 1; j <= n; ++j) {
		temp = bmat[kold + j * bmat_dim1];
		bmat[kold + j * bmat_dim1] = bmat[knew + j * bmat_dim1];
		bmat[knew + j * bmat_dim1] = temp;
	}
	for (long j = 1; j <= nptm; ++j) {
		temp = zmat[kold + j * zmat_dim1];
		zmat[kold + j * zmat_dim1] = zmat[knew + j * zmat_dim1];
		zmat[knew + j * zmat_dim1] = temp;
	}
	ptsid[kold] = ptsid[knew];
	ptsid[knew] = 0.;
	w[ndim + knew] = 0.;
	--nrem;
	if (knew != kopt) {
		temp = vlag[kold];
		vlag[kold] = vlag[knew];
		vlag[knew] = temp;

		/*     Update the BMAT and ZMAT matrices so that the status of the KNEW-th */
		/*     interpolation point can be changed from provisional to original. The */
		/*     branch to label 350 occurs if all the original points are reinstated. */
		/*     The nonnegative values of W(NDIM+K) are required in the search below. */
		update(n, npt, bmat + bmat_offset, zmat + zmat_offset, ndim, vlag, beta,
				denom, knew, w);
		if (nrem == 0) {
			goto L350;
		}
		for (long k = 1; k <= npt; ++k) {
			w[ndim + k] = std::abs(w[ndim + k]);
		}
	}

	/*     Pick the index KNEW of an original interpolation point that has not */
	/*     yet replaced one of the provisional interpolation points, giving */
	/*     attention to the closeness to XOPT and to previous tries with KNEW. */
	L120: dsqmin = 0.;
	for (long k = 1; k <= npt; ++k) {
		if (w[ndim + k] > 0.) {
			if (dsqmin == 0. || w[ndim + k] < dsqmin) {
				knew = k;
				dsqmin = w[ndim + k];
			}
		}
	}
	if (dsqmin == 0.0) {
		goto L260;
	}

	/*     Form the W-vector of the chosen original interpolation point. */
	for (long j = 1; j <= n; ++j) {
		w[npt + j] = xpt[knew + j * xpt_dim1];
	}
	for (long k = 1; k <= npt; ++k) {
		sum = 0.;
		if (k == kopt) {
		} else if (ptsid[k] == 0.) {
			for (long j = 1; j <= n; ++j) {
				sum += w[npt + j] * xpt[k + j * xpt_dim1];
			}
		} else {
			const long ip = long(ptsid[k]);
			if (ip > 0) {
				sum = w[npt + ip] * ptsaux[(ip << 1) + 1];
			}
			iq = long(double(np) * ptsid[k] - double(ip * np));
			if (iq > 0) {
				iw = 1;
				if (ip == 0) {
					iw = 2;
				}
				sum += w[npt + iq] * ptsaux[iw + (iq << 1)];
			}
		}
		w[k] = 0.5 * sum * sum;
	}

	/*     Calculate VLAG and BETA for the required updating of the H matrix if */
	/*     XPT(KNEW,.) is reinstated in the set of interpolation points. */
	for (long k = 1; k <= npt; ++k) {
		sum = 0.;
		for (long j = 1; j <= n; ++j) {
			sum += bmat[k + j * bmat_dim1] * w[npt + j];
		}
		vlag[k] = sum;
	}
	beta = 0.;
	for (long j = 1; j <= nptm; ++j) {
		sum = 0.;
		for (long k = 1; k <= npt; ++k) {
			sum += zmat[k + j * zmat_dim1] * w[k];
		}
		beta -= sum * sum;
		for (long k = 1; k <= npt; ++k) {
			vlag[k] += sum * zmat[k + j * zmat_dim1];
		}
	}
	bsum = 0.;
	distsq = 0.;
	for (long j = 1; j <= n; ++j) {
		sum = 0.;
		for (long k = 1; k <= npt; ++k) {
			sum += bmat[k + j * bmat_dim1] * w[k];
		}
		jp = j + npt;
		bsum += sum * w[jp];
		for (long ip = npt + 1; ip <= ndim; ++ip) {
			sum += bmat[ip + j * bmat_dim1] * w[ip];
		}
		bsum += sum * w[jp];
		vlag[jp] = sum;
		distsq += std::pow(xpt[knew + j * xpt_dim1], 2);
	}
	beta = 0.5 * distsq * distsq + beta - bsum;
	vlag[kopt] += 1.;

	/*     KOLD is set to the index of the provisional interpolation point that is */
	/*     going to be deleted to make way for the KNEW-th original interpolation */
	/*     point. The choice of KOLD is governed by the avoidance of a small value */
	/*     of the denominator in the updating calculation of UPDATE. */
	denom = 0.;
	vlmxsq = 0.;
	for (long k = 1; k <= npt; ++k) {
		if (ptsid[k] != 0.) {
			hdiag = 0.;
			for (long j = 1; j <= nptm; ++j) {
				hdiag += std::pow(zmat[k + j * zmat_dim1], 2);
			}
			den = beta * hdiag + vlag[k] * vlag[k];
			if (den > denom) {
				kold = k;
				denom = den;
			}
		}
		vlmxsq = std::max(vlmxsq, vlag[k] * vlag[k]);
	}
	if (denom <= vlmxsq * 0.01) {
		w[ndim + knew] = -w[ndim + knew] - winc;
		goto L120;
	}
	goto L80;

	/*     When label 260 is reached, all the final positions of the interpolation */
	/*     points have been chosen although any changes have not been included yet */
	/*     in XPT. Also the final BMAT and ZMAT matrices are complete, but, apart */
	/*     from the shift of XBASE, the updating of the quadratic model remains to */
	/*     be done. The following cycle through the new interpolation points begins */
	/*     by putting the new point in XPT(KPT,.) and by setting PQ(KPT) to zero, */
	/*     except that a RETURN occurs if MAXFUN prohibits another value of F. */
	L260: for (long kpt = 1; kpt <= npt; ++kpt) {
		if (ptsid[kpt] == 0.) {
			goto L340;
		}
		if (nf >= maxfun) {
			nf = -1;
			goto L350;
		}
		ih = 0;
		for (long j = 1; j <= n; ++j) {
			w[j] = xpt[kpt + j * xpt_dim1];
			xpt[kpt + j * xpt_dim1] = 0.;
			temp = pq[kpt] * w[j];
			for (long i = 1; i <= j; ++i) {
				++ih;
				hq[ih] += temp * w[i];
			}
		}
		pq[kpt] = 0.;
		{
			const long ip = long(ptsid[kpt]);
			iq = long(double(np) * ptsid[kpt] - double(ip * np));
			if (ip > 0) {
				xp = ptsaux[(ip << 1) + 1];
				xpt[kpt + ip * xpt_dim1] = xp;
			}
			if (iq > 0) {
				xq = ptsaux[(iq << 1) + 1];
				if (ip == 0) {
					xq = ptsaux[(iq << 1) + 2];
				}
				xpt[kpt + iq * xpt_dim1] = xq;
			}

			/*     Set VQUAD to the value of the current model at the new point. */
			vquad = fbase;
			if (ip > 0) {
				ihp = (ip + ip * ip) / 2;
				vquad += xp * (gopt[ip] + 0.5 * xp * hq[ihp]);
			}
			if (iq > 0) {
				ihq = (iq + iq * iq) / 2;
				vquad += xq * (gopt[iq] + 0.5 * xq * hq[ihq]);
				if (ip > 0) {
					iw = std::max(ihp, ihq) - std::abs(ip - iq);
					vquad += xp * xq * hq[iw];
				}
			}
			for (long k = 1; k <= npt; ++k) {
				temp = 0.;
				if (ip > 0) {
					temp += xp * xpt[k + ip * xpt_dim1];
				}
				if (iq > 0) {
					temp += xq * xpt[k + iq * xpt_dim1];
				}
				vquad += 0.5 * pq[k] * temp * temp;
			}
		}

		/*     Calculate F at the new interpolation point, and set DIFF to the factor */
		/*     that is going to multiply the KPT-th Lagrange function when the model */
		/*     is updated to provide interpolation to the new function value. */
		for (long i = 1; i <= n; ++i) {
			w[i] = std::min(std::max(xl[i], xbase[i] + xpt[kpt + i * xpt_dim1]),
					xu[i]);
			if (xpt[kpt + i * xpt_dim1] == sl[i]) {
				w[i] = xl[i];
			}
			if (xpt[kpt + i * xpt_dim1] == su[i]) {
				w[i] = xu[i];
			}
		}
		++nf;
		{
			const double f = function(w + 1);
			fev++;
			fval[kpt] = f;
			if (f < fval[kopt]) {
				kopt = kpt;
			}
			diff = f - vquad;
		}

		/*     Update the quadratic model. The RETURN from the subroutine occurs when */
		/*     all the new interpolation points are included in the model. */
		for (long i = 1; i <= n; ++i) {
			gopt[i] += diff * bmat[kpt + i * bmat_dim1];
		}
		for (long k = 1; k <= npt; ++k) {
			sum = 0.;
			for (long j = 1; j <= nptm; ++j) {
				sum += zmat[k + j * zmat_dim1] * zmat[kpt + j * zmat_dim1];
			}
			temp = diff * sum;
			if (ptsid[k] == 0.) {
				pq[k] += temp;
			} else {
				const long ip = long(ptsid[k]);
				iq = long(double(np) * ptsid[k] - double(ip * np));
				ihq = (iq * iq + iq) / 2;
				if (ip == 0) {
					hq[ihq] += temp * std::pow(ptsaux[(iq << 1) + 2], 2);
				} else {
					ihp = (ip * ip + ip) / 2;
					hq[ihp] += temp * std::pow(ptsaux[(ip << 1) + 1], 2);
					if (iq > 0) {
						hq[ihq] += temp * std::pow(ptsaux[(iq << 1) + 1], 2);
						iw = std::max(ihp, ihq) - std::abs(iq - ip);
						hq[iw] += temp * ptsaux[(ip << 1) + 1]
								* ptsaux[(iq << 1) + 1];
					}
				}
			}
		}
		ptsid[kpt] = 0.;
		L340: ;
	}
	L350: ;
}

void Bobyqa::trsbox(const long n, const long npt, const double *xpt,
		const double *const xopt, const double *const gopt,
		const double *const hq, const double *const pq, const double *const sl,
		const double *const su, const double delta, double *const xnew,
		double *const d, double *const gnew, double *const xbdi,
		double *const s, double *const hs, double *const hred,
		double *const dsq, double *const crvmin) {

	/* Local variables */
	double ds;
	long iu;
	double dhd, dhs, cth, shs, sth, ssq, beta, sdec, blen;
	long iact = 0, nact = 0;
	double angt, qred;
	long isav;
	double temp = 0, xsav = 0, xsum = 0, angbd = 0, dredg = 0, sredg = 0;
	long iterc;
	double resid = 0, delsq = 0, ggsav = 0, tempa = 0, tempb = 0, redmax = 0,
			dredsq = 0, redsav = 0, gredsq = 0, rednew = 0;
	long itcsav = 0;
	double rdprev = 0, rdnext = 0, stplen = 0, stepsq = 0;
	long itermax = 0;

	/*     The arguments N, NPT, XPT, XOPT, GOPT, HQ, PQ, SL and SU have the same */
	/*       meanings as the corresponding arguments of BOBYQB. */
	/*     DELTA is the trust region radius for the present calculation, which */
	/*       seeks a small value of the quadratic model within distance DELTA of */
	/*       XOPT subject to the bounds on the variables. */
	/*     XNEW will be set to a new vector of variables that is approximately */
	/*       the one that minimizes the quadratic model within the trust region */
	/*       subject to the SL and SU constraints on the variables. It satisfies */
	/*       as equations the bounds that become active during the calculation. */
	/*     D is the calculated trial step from XOPT, generated iteratively from an */
	/*       initial value of zero. Thus XNEW is XOPT+D after the final iteration. */
	/*     GNEW holds the gradient of the quadratic model at XOPT+D. It is updated */
	/*       when D is updated. */
	/*     XBDI is a working space vector. For I=1,2,...,N, the element XBDI(I) is */
	/*       set to -1.0, 0.0, or 1.0, the value being nonzero if and only if the */
	/*       I-th variable has become fixed at a bound, the bound being SL(I) or */
	/*       SU(I) in the case XBDI(I)=-1.0 or XBDI(I)=1.0, respectively. This */
	/*       information is accumulated during the construction of XNEW. */
	/*     The arrays S, HS and HRED are also used for working space. They hold the */
	/*       current search direction, and the changes in the gradient of Q along S */
	/*       and the reduced D, respectively, where the reduced D is the same as D, */
	/*       except that the components of the fixed variables are zero. */
	/*     DSQ will be set to the square of the length of XNEW-XOPT. */
	/*     CRVMIN is set to zero if D reaches the trust region boundary. Otherwise */
	/*       it is set to the least curvature of H that occurs in the conjugate */
	/*       gradient searches that are not restricted by any constraints. The */
	/*       value CRVMIN=-1.0D0 is set, however, if all of these searches are */
	/*       constrained. */

	/*     A version of the truncated conjugate gradient is applied. If a line */
	/*     search is restricted by a constraint, then the procedure is restarted, */
	/*     the values of the variables that are at their bounds being fixed. If */
	/*     the trust region boundary is reached, then further changes may be made */
	/*     to D, each one being in the two dimensional space that is spanned */
	/*     by the current D and the gradient of Q at XOPT+D, staying on the trust */
	/*     region boundary. Termination occurs when the reduction in Q seems to */
	/*     be close to the greatest reduction that can be achieved. */

	/*     Set some constants. */

	/* Parameter adjustments */
	const long xpt_dim1 = npt;
	const long xpt_offset = 1 + xpt_dim1;
	xpt -= xpt_offset;

	/* Function Body */

	/*     The sign of GOPT(I) gives the sign of the change to the I-th variable */
	/*     that will reduce Q from its value at XOPT. Thus XBDI(I) shows whether */
	/*     or not to fix the I-th variable at one of its bounds initially, with */
	/*     NACT being set to the number of fixed variables. D and GNEW are also */
	/*     set for the first iteration. DELSQ is the upper bound on the sum of */
	/*     squares of the free variables. QRED is the reduction in Q so far. */
	iterc = 0;
	nact = 0;
	for (long i = 1; i <= n; ++i) {
		xbdi[i] = 0.;
		if (xopt[i] <= sl[i]) {
			if (gopt[i] >= 0.) {
				xbdi[i] = -1.;
			}
		} else if (xopt[i] >= su[i]) {
			if (gopt[i] <= 0.) {
				xbdi[i] = 1.;
			}
		}
		if (xbdi[i] != 0.) {
			++nact;
		}
		d[i] = 0.;
		gnew[i] = gopt[i];
	}
	delsq = delta * delta;
	qred = 0.;
	*crvmin = -1.;

	/*     Set the next search direction of the conjugate gradient method. It is */
	/*     the steepest descent direction initially and when the iterations are */
	/*     restarted because a variable has just been fixed by a bound, and of */
	/*     course the components of the fixed variables are zero. ITERMAX is an */
	/*     upper bound on the indices of the conjugate gradient iterations. */
	L20: beta = 0.;
	L30: stepsq = 0.;
	for (long i = 1; i <= n; ++i) {
		if (xbdi[i] != 0.) {
			s[i] = 0.;
		} else if (beta == 0.) {
			s[i] = -gnew[i];
		} else {
			s[i] = beta * s[i] - gnew[i];
		}
		stepsq += s[i] * s[i];
	}
	if (stepsq == 0.) {
		goto L190;
	}
	if (beta == 0.) {
		gredsq = stepsq;
		itermax = iterc + n - nact;
	}
	if (gredsq * delsq <= qred * 1e-4 * qred) {
		goto L190;
	}

	/*     Multiply the search direction by the second derivative matrix of Q and */
	/*     calculate some scalars for the choice of steplength. Then set BLEN to */
	/*     the length of the the step to the trust region boundary and STPLEN to */
	/*     the steplength, ignoring the simple bounds. */
	goto L210;
	L50: resid = delsq;
	ds = 0.;
	shs = 0.;
	for (long i = 1; i <= n; ++i) {
		if (xbdi[i] == 0.) {
			resid -= d[i] * d[i];
			ds += s[i] * d[i];
			shs += s[i] * hs[i];
		}
	}
	if (resid <= 0.) {
		goto L90;
	}
	temp = std::sqrt(stepsq * resid + ds * ds);
	if (ds < 0.) {
		blen = (temp - ds) / stepsq;
	} else {
		blen = resid / (temp + ds);
	}
	stplen = blen;
	if (shs > 0.) {
		stplen = std::min(blen, gredsq / shs);
	}

	/*     Reduce STPLEN if necessary in order to preserve the simple bounds, */
	/*     letting IACT be the index of the new constrained variable. */
	iact = 0;
	for (long i = 1; i <= n; ++i) {
		if (s[i] != 0.) {
			xsum = xopt[i] + d[i];
			if (s[i] > 0.) {
				temp = (su[i] - xsum) / s[i];
			} else {
				temp = (sl[i] - xsum) / s[i];
			}
			if (temp < stplen) {
				stplen = temp;
				iact = i;
			}
		}
	}

	/*     Update CRVMIN, GNEW and D. Set SDEC to the decrease that occurs in Q. */
	sdec = 0.;
	if (stplen > 0.) {
		++iterc;
		temp = shs / stepsq;
		if (iact == 0 && temp > 0.) {
			*crvmin = std::min(*crvmin, temp);
			if (*crvmin == -1.) {
				*crvmin = temp;
			}
		}
		ggsav = gredsq;
		gredsq = 0.;
		for (long i = 1; i <= n; ++i) {
			gnew[i] += stplen * hs[i];
			if (xbdi[i] == 0.) {
				gredsq += gnew[i] * gnew[i];
			}
			d[i] += stplen * s[i];
		}
		sdec = std::max(stplen * (ggsav - 0.5 * stplen * shs), 0.);
		qred += sdec;
	}

	/*     Restart the conjugate gradient method if it has hit a new bound. */
	if (iact > 0) {
		++nact;
		xbdi[iact] = 1.;
		if (s[iact] < 0.) {
			xbdi[iact] = -1.;
		}
		delsq -= d[iact] * d[iact];
		if (delsq <= 0.) {
			goto L90;
		}
		goto L20;
	}

	/*     If STPLEN is less than BLEN, then either apply another conjugate */
	/*     gradient iteration or RETURN. */
	if (stplen < blen) {
		if (iterc == itermax) {
			goto L190;
		}
		if (sdec <= qred * 0.01) {
			goto L190;
		}
		beta = gredsq / ggsav;
		goto L30;
	}
	L90: *crvmin = 0.;

	/*     Prepare for the alternative iteration by calculating some scalars */
	/*     and by multiplying the reduced D by the second derivative matrix of */
	/*     Q, where S holds the reduced D in the call of GGMULT. */
	L100: if (nact >= n - 1) {
		goto L190;
	}
	dredsq = 0.;
	dredg = 0.;
	gredsq = 0.;
	for (long i = 1; i <= n; ++i) {
		if (xbdi[i] == 0.) {
			dredsq += d[i] * d[i];
			dredg += d[i] * gnew[i];
			gredsq += gnew[i] * gnew[i];
			s[i] = d[i];
		} else {
			s[i] = 0.;
		}
	}
	itcsav = iterc;
	goto L210;

	/*     Let the search direction S be a linear combination of the reduced D */
	/*     and the reduced G that is orthogonal to the reduced D. */
	L120: ++iterc;
	temp = gredsq * dredsq - dredg * dredg;
	if (temp <= qred * 1e-4 * qred) {
		goto L190;
	}
	temp = std::sqrt(temp);
	for (long i = 1; i <= n; ++i) {
		if (xbdi[i] == 0.) {
			s[i] = (dredg * d[i] - dredsq * gnew[i]) / temp;
		} else {
			s[i] = 0.;
		}
	}
	sredg = -temp;

	/*     By considering the simple bounds on the variables, calculate an upper */
	/*     bound on the tangent of half the angle of the alternative iteration, */
	/*     namely ANGBD, except that, if already a free variable has reached a */
	/*     bound, there is a branch back to label 100 after fixing that variable. */
	angbd = 1.;
	iact = 0;
	for (long i = 1; i <= n; ++i) {
		if (xbdi[i] == 0.) {
			tempa = xopt[i] + d[i] - sl[i];
			tempb = su[i] - xopt[i] - d[i];
			if (tempa <= 0.) {
				++nact;
				xbdi[i] = -1.;
				goto L100;
			} else if (tempb <= 0.) {
				++nact;
				xbdi[i] = 1.;
				goto L100;
			}
			ssq = d[i] * d[i] + s[i] * s[i];
			temp = ssq - std::pow(xopt[i] - sl[i], 2);
			if (temp > 0.) {
				temp = std::sqrt(temp) - s[i];
				if (angbd * temp > tempa) {
					angbd = tempa / temp;
					iact = i;
					xsav = -1.;
				}
			}
			temp = ssq - std::pow(su[i] - xopt[i], 2);
			if (temp > 0.) {
				temp = std::sqrt(temp) + s[i];
				if (angbd * temp > tempb) {
					angbd = tempb / temp;
					iact = i;
					xsav = 1.;
				}
			}
		}
	}

	/*     Calculate HHD and some curvatures for the alternative iteration. */
	goto L210;
	L150: shs = 0.;
	dhs = 0.;
	dhd = 0.;
	for (long i = 1; i <= n; ++i) {
		if (xbdi[i] == 0.) {
			shs += s[i] * hs[i];
			dhs += d[i] * hs[i];
			dhd += d[i] * hred[i];
		}
	}

	/*     Seek the greatest reduction in Q for a range of equally spaced values */
	/*     of ANGT in [0,ANGBD], where ANGT is the tangent of half the angle of */
	/*     the alternative iteration. */
	redmax = 0.;
	isav = 0;
	redsav = 0.;
	iu = long(angbd * 17. + 3.1);
	for (long i = 1; i <= iu; ++i) {
		angt = angbd * double(i) / double(iu);
		sth = (angt + angt) / (1. + angt * angt);
		temp = shs + angt * (angt * dhd - dhs - dhs);
		rednew = sth * (angt * dredg - sredg - 0.5 * sth * temp);
		if (rednew > redmax) {
			redmax = rednew;
			isav = i;
			rdprev = redsav;
		} else if (i == isav + 1) {
			rdnext = rednew;
		}
		redsav = rednew;
	}

	/*     Return if the reduction is zero. Otherwise, set the sine and cosine */
	/*     of the angle of the alternative iteration, and calculate SDEC. */
	if (isav == 0) {
		goto L190;
	}
	if (isav < iu) {
		temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext);
		angt = angbd * (double(isav) + 0.5 * temp) / double(iu);
	}
	cth = (1. - angt * angt) / (1. + angt * angt);
	sth = (angt + angt) / (1. + angt * angt);
	temp = shs + angt * (angt * dhd - dhs - dhs);
	sdec = sth * (angt * dredg - sredg - 0.5 * sth * temp);
	if (sdec <= 0.) {
		goto L190;
	}

	/*     Update GNEW, D and HRED. If the angle of the alternative iteration */
	/*     is restricted by a bound on a free variable, that variable is fixed */
	/*     at the bound. */
	dredg = 0.;
	gredsq = 0.;
	for (long i = 1; i <= n; ++i) {
		gnew[i] = gnew[i] + (cth - 1.) * hred[i] + sth * hs[i];
		if (xbdi[i] == 0.) {
			d[i] = cth * d[i] + sth * s[i];
			dredg += d[i] * gnew[i];
			gredsq += gnew[i] * gnew[i];
		}
		hred[i] = cth * hred[i] + sth * hs[i];
	}
	qred += sdec;
	if (iact > 0 && isav == iu) {
		++nact;
		xbdi[iact] = xsav;
		goto L100;
	}

	/*     If SDEC is sufficiently small, then RETURN after setting XNEW to */
	/*     XOPT+D, giving careful attention to the bounds. */
	if (sdec > qred * 0.01) {
		goto L120;
	}
	L190: *dsq = 0.;
	for (long i = 1; i <= n; ++i) {
		xnew[i] = std::max(std::min(xopt[i] + d[i], su[i]), sl[i]);
		if (xbdi[i] == -1.) {
			xnew[i] = sl[i];
		}
		if (xbdi[i] == 1.) {
			xnew[i] = su[i];
		}
		d[i] = xnew[i] - xopt[i];
		*dsq += d[i] * d[i];
	}
	return;

	/*     The following instructions multiply the current S-vector by the second */
	/*     derivative matrix of the quadratic model, putting the product in HS. */
	/*     They are reached from three different parts of the software above and */
	/*     they can be regarded as an external subroutine. */
	L210: long ih = 0;
	for (long j = 1; j <= n; ++j) {
		hs[j] = 0.;
		for (long i = 1; i <= j; ++i) {
			++ih;
			if (i < j) {
				hs[j] += hq[ih] * s[i];
			}
			hs[i] += hq[ih] * s[j];
		}
	}
	for (long k = 1; k <= npt; ++k) {
		if (pq[k] != 0.) {
			temp = 0.;
			for (long j = 1; j <= n; ++j) {
				temp += xpt[k + j * xpt_dim1] * s[j];
			}
			temp *= pq[k];
			for (long i = 1; i <= n; ++i) {
				hs[i] += temp * xpt[k + i * xpt_dim1];
			}
		}
	}
	if (*crvmin != 0.) {
		goto L50;
	}
	if (iterc > itcsav) {
		goto L150;
	}
	for (long i = 1; i <= n; ++i) {
		hred[i] = hs[i];
	}
	goto L120;
}

double less_abs(const double lhs, const double rhs) {
	return std::abs(lhs) < std::abs(rhs);
}

void Bobyqa::update(const long n, const long npt, double *bmat, double *zmat,
		const long ndim, double *const vlag, const double beta,
		const double denom, const long knew, double *const w) {

	/*     The arrays BMAT and ZMAT are updated, as required by the new position */
	/*     of the interpolation point that has the index KNEW. The vector VLAG has */
	/*     N+NPT components, set on entry to the first NPT and last N components */
	/*     of the product Hw in equation (4.11) of the Powell (2006) paper on */
	/*     NEWUOA. Further, BETA is set on entry to the value of the parameter */
	/*     with that name, and DENOM is set to the denominator of the updating */
	/*     formula. Elements of ZMAT may be treated as zero if their moduli are */
	/*     at most ZTEST. The first NDIM elements of W are used for working space. */

	/*     Set some constants. */

	/* Parameter adjustments */
	const long zmat_dim1 = npt;
	const long zmat_offset = 1 + zmat_dim1;
	zmat -= zmat_offset;
	const long bmat_dim1 = ndim;
	const long bmat_offset = 1 + bmat_dim1;
	bmat -= bmat_offset;

	/* Function Body */
	const long nptm = npt - n - 1;
	const auto zmat_end = zmat + zmat_offset + nptm * npt;
	const auto zmat_max = std::max_element(zmat + zmat_offset, zmat_end,
			less_abs);
	const double ztest = zmat_max == zmat_end ? 0 : *zmat_max * 1e-20;

	/*     Apply the rotations that put zeros in the KNEW-th row of ZMAT. */
	for (long j = 2; j <= nptm; ++j) {
		if (std::abs(zmat[knew + j * zmat_dim1]) > ztest) {
			const double temp = std::hypot(zmat[knew + zmat_dim1],
					zmat[knew + j * zmat_dim1]);
			const double tempa = zmat[knew + zmat_dim1] / temp;
			const double tempb = zmat[knew + j * zmat_dim1] / temp;
			for (long i = 1; i <= npt; ++i) {
				const double temp = tempa * zmat[i + zmat_dim1]
						+ tempb * zmat[i + j * zmat_dim1];
				zmat[i + j * zmat_dim1] = tempa * zmat[i + j * zmat_dim1]
						- tempb * zmat[i + zmat_dim1];
				zmat[i + zmat_dim1] = temp;
			}
		}
		zmat[knew + j * zmat_dim1] = 0.;
	}

	/*     Put the first NPT components of the KNEW-th column of HLAG into W, */
	/*     and calculate the parameters of the updating formula. */
	for (long i = 1; i <= npt; ++i) {
		w[i] = zmat[knew + zmat_dim1] * zmat[i + zmat_dim1];
	}
	const double alpha = w[knew];
	const double tau = vlag[knew];
	vlag[knew] -= 1.;

	/*     Complete the updating of ZMAT. */
	const double temp = std::sqrt(denom);
	const double tempb = zmat[knew + zmat_dim1] / temp;
	const double tempa = tau / temp;
	for (long i = 1; i <= npt; ++i) {
		zmat[i + zmat_dim1] = tempa * zmat[i + zmat_dim1] - tempb * vlag[i];
	}

	/*     Finally, update the matrix BMAT. */
	for (long j = 1; j <= n; ++j) {
		const long jp = npt + j;
		w[jp] = bmat[knew + j * bmat_dim1];
		const double tempa = (alpha * vlag[jp] - tau * w[jp]) / denom;
		const double tempb = (-(beta) * w[jp] - tau * vlag[jp]) / denom;
		for (long i = 1; i <= jp; ++i) {
			bmat[i + j * bmat_dim1] = bmat[i + j * bmat_dim1] + tempa * vlag[i]
					+ tempb * w[i];
			if (i > npt) {
				bmat[jp + (i - npt) * bmat_dim1] = bmat[i + j * bmat_dim1];
			}
		}
	}
}

double Bobyqa::bobyqb(const multivariate &function, const long n,
		const long npt, double *const x, const double *const xl,
		const double *const xu, const double rhobeg, const double rhoend,
		const long maxfun, double *const xbase, double *xpt, double *const fval,
		double *const xopt, double *const gopt, double *const hq,
		double *const pq, double *bmat, double *zmat, const long ndim,
		double *const sl, double *const su, double *const xnew,
		double *const xalt, double *const d, double *const vlag,
		double *const w, long &fev) {

	/* Local variables */
	double f = 0;
	long ih, nf, jp;
	double dx;
	double den = 0, dsq = 0, rho = 0, sum = 0, diff = 0, beta = 0, gisq = 0;
	long knew = 0;
	double temp, suma, sumb, bsum, fopt;
	long kopt = 0;
	double curv;
	long ksav;
	double gqsq = 0, dist = 0, sumw = 0, sumz = 0, diffa = 0, diffb = 0, diffc =
			0, hdiag = 0;
	long kbase;
	double alpha = 0, delta = 0, adelt = 0, denom = 0, fsave = 0, bdtol = 0,
			delsq = 0;
	long nresc, nfsav;
	double ratio = 0, dnorm = 0, vquad = 0, pqold = 0;
	long itest;
	double sumpq, scaden;
	double errbig, cauchy = 0, fracsq, biglsq, densav;
	double bdtest;
	double crvmin, frhosq;
	double distsq;
	long ntrits;
	double xoptsq;

	/*     The arguments N, NPT, X, XL, XU, RHOBEG, RHOEND, IPRINT and MAXFUN */
	/*       are identical to the corresponding arguments in SUBROUTINE BOBYQA. */
	/*     XBASE holds a shift of origin that should reduce the contributions */
	/*       from rounding errors to values of the model and Lagrange functions. */
	/*     XPT is a two-dimensional array that holds the coordinates of the */
	/*       interpolation points relative to XBASE. */
	/*     FVAL holds the values of F at the interpolation points. */
	/*     XOPT is set to the displacement from XBASE of the trust region centre. */
	/*     GOPT holds the gradient of the quadratic model at XBASE+XOPT. */
	/*     HQ holds the explicit second derivatives of the quadratic model. */
	/*     PQ contains the parameters of the implicit second derivatives of the */
	/*       quadratic model. */
	/*     BMAT holds the last N columns of H. */
	/*     ZMAT holds the factorization of the leading NPT by NPT submatrix of H, */
	/*       this factorization being ZMAT times ZMAT^T, which provides both the */
	/*       correct rank and positive semi-definiteness. */
	/*     NDIM is the first dimension of BMAT and has the value NPT+N. */
	/*     SL and SU hold the differences XL-XBASE and XU-XBASE, respectively. */
	/*       All the components of every XOPT are going to satisfy the bounds */
	/*       SL(I) .LEQ. XOPT(I) .LEQ. SU(I), with appropriate equalities when */
	/*       XOPT is on a constraint boundary. */
	/*     XNEW is chosen by SUBROUTINE TRSBOX or ALTMOV. Usually XBASE+XNEW is the */
	/*       vector of variables for the next call of CALFUN. XNEW also satisfies */
	/*       the SL and SU constraints in the way that has just been mentioned. */
	/*     XALT is an alternative to XNEW, chosen by ALTMOV, that may replace XNEW */
	/*       in order to increase the denominator in the updating of UPDATE. */
	/*     D is reserved for a trial step from XOPT, which is usually XNEW-XOPT. */
	/*     VLAG contains the values of the Lagrange functions at a new point X. */
	/*       They are part of a product that requires VLAG to be of length NDIM. */
	/*     W is a one-dimensional array that is used for working space. Its length */
	/*       must be at least 3*NDIM = 3*(NPT+N). */

	/*     Set some constants. */

	/* Parameter adjustments */
	const long zmat_dim1 = npt;
	const long zmat_offset = 1 + zmat_dim1;
	zmat -= zmat_offset;
	const long xpt_dim1 = npt;
	const long xpt_offset = 1 + xpt_dim1;
	xpt -= xpt_offset;
	const long bmat_dim1 = ndim;
	const long bmat_offset = 1 + bmat_dim1;
	bmat -= bmat_offset;

	/* Function Body */
	const long np = n + 1;
	const long nptm = npt - np;
	const long nh = n * np / 2;

	/*     The call of PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ, */
	/*     BMAT and ZMAT for the first iteration, with the corresponding values of */
	/*     of NF and KOPT, which are the number of calls of CALFUN so far and the */
	/*     index of the interpolation point at the trust region centre. Then the */
	/*     initial XOPT is set too. The branch to label 720 occurs if MAXFUN is */
	/*     less than NPT. GOPT will be updated if KOPT is different from KBASE. */
	prelim(function, n, npt, x, xl, xu, rhobeg, maxfun, xbase, xpt + xpt_offset,
			fval, gopt, hq, pq, bmat + bmat_offset, zmat + zmat_offset, ndim,
			sl, su, nf, kopt, fev);
	xoptsq = 0.;
	for (long i = 1; i <= n; ++i) {
		xopt[i] = xpt[kopt + i * xpt_dim1];
		xoptsq += xopt[i] * xopt[i];
	}
	fsave = fval[1];
	if (nf < npt) {
		goto L720;
	}
	kbase = 1;

	/*     Complete the settings that are required for the iterative procedure. */
	rho = rhobeg;
	delta = rho;
	nresc = nf;
	ntrits = 0;
	diffa = 0.;
	diffb = 0.;
	itest = 0;
	nfsav = nf;

	/*     Update GOPT if necessary before the first iteration and after each */
	/*     call of RESCUE that makes a call of CALFUN. */
	L20: if (kopt != kbase) {
		ih = 0;
		for (long j = 1; j <= n; ++j) {
			for (long i = 1; i <= j; ++i) {
				++ih;
				if (i < j) {
					gopt[j] += hq[ih] * xopt[i];
				}
				gopt[i] += hq[ih] * xopt[j];
			}
		}
		if (nf > npt) {
			for (long k = 1; k <= npt; ++k) {
				temp = 0.;
				for (long j = 1; j <= n; ++j) {
					temp += xpt[k + j * xpt_dim1] * xopt[j];
				}
				temp = pq[k] * temp;
				for (long i = 1; i <= n; ++i) {
					gopt[i] += temp * xpt[k + i * xpt_dim1];
				}
			}
		}
	}

	/*     Generate the next point in the trust region that provides a small value */
	/*     of the quadratic model subject to the constraints on the variables. */
	/*     The long NTRITS is set to the number "trust region" iterations that */
	/*     have occurred since the last "alternative" iteration. If the length */
	/*     of XNEW-XOPT is less than HALF*RHO, however, then there is a branch to */
	/*     label 650 or 680 with NTRITS=-1, instead of calculating F at XNEW. */
	L60: trsbox(n, npt, xpt + xpt_offset, xopt, gopt, hq, pq, sl, su, delta,
			xnew, d, w, w + np - 1, w + np + n - 1, w + np + (n << 1) - 1,
			w + np + n * 3 - 1, &dsq, &crvmin);
	dnorm = std::min(delta, std::sqrt(dsq));
	if (dnorm < 0.5 * rho) {
		ntrits = -1;
		distsq = std::pow(10. * rho, 2);
		if (nf <= nfsav + 2) {
			goto L650;
		}

		/*     The following choice between labels 650 and 680 depends on whether or */
		/*     not our work with the current RHO seems to be complete. Either RHO is */
		/*     decreased or termination occurs if the errors in the quadratic model at */
		/*     the last three interpolation points compare favourably with predictions */
		/*     of likely improvements to the model within distance HALF*RHO of XOPT. */
		errbig = std::max(std::max(diffa, diffb), diffc);
		frhosq = rho * 0.125 * rho;
		if (crvmin > 0. && errbig > frhosq * crvmin) {
			goto L650;
		}
		bdtol = errbig / rho;
		for (long j = 1; j <= n; ++j) {
			bdtest = bdtol;
			if (xnew[j] == sl[j]) {
				bdtest = w[j];
			}
			if (xnew[j] == su[j]) {
				bdtest = -w[j];
			}
			if (bdtest < bdtol) {
				curv = hq[(j + j * j) / 2];
				for (long k = 1; k <= npt; ++k) {
					curv += pq[k] * std::pow(xpt[k + j * xpt_dim1], 2);
				}
				bdtest += 0.5 * curv * rho;
				if (bdtest < bdtol) {
					goto L650;
				}
			}
		}
		goto L680;
	}
	++ntrits;

	/*     Severe cancellation is likely to occur if XOPT is too far from XBASE. */
	/*     If the following test holds, then XBASE is shifted so that XOPT becomes */
	/*     zero. The appropriate changes are made to BMAT and to the second */
	/*     derivatives of the current model, beginning with the changes to BMAT */
	/*     that do not depend on ZMAT. VLAG is used temporarily for working space. */

	L90: if (dsq <= xoptsq * 0.001) {
		fracsq = xoptsq * 0.25;
		sumpq = 0.;
		for (long k = 1; k <= npt; ++k) {
			sumpq += pq[k];
			sum = -0.5 * xoptsq;
			for (long i = 1; i <= n; ++i) {
				sum += xpt[k + i * xpt_dim1] * xopt[i];
			}
			w[npt + k] = sum;
			temp = fracsq - 0.5 * sum;
			for (long i = 1; i <= n; ++i) {
				w[i] = bmat[k + i * bmat_dim1];
				vlag[i] = sum * xpt[k + i * xpt_dim1] + temp * xopt[i];
				const long ip = npt + i;
				for (long j = 1; j <= i; ++j) {
					bmat[ip + j * bmat_dim1] = bmat[ip + j * bmat_dim1]
							+ w[i] * vlag[j] + vlag[i] * w[j];
				}
			}
		}

		/*     Then the revisions of BMAT that depend on ZMAT are calculated. */
		for (long jj = 1; jj <= nptm; ++jj) {
			sumz = 0.;
			sumw = 0.;
			for (long k = 1; k <= npt; ++k) {
				sumz += zmat[k + jj * zmat_dim1];
				vlag[k] = w[npt + k] * zmat[k + jj * zmat_dim1];
				sumw += vlag[k];
			}
			for (long j = 1; j <= n; ++j) {
				sum = (fracsq * sumz - 0.5 * sumw) * xopt[j];
				for (long k = 1; k <= npt; ++k) {
					sum += vlag[k] * xpt[k + j * xpt_dim1];
				}
				w[j] = sum;
				for (long k = 1; k <= npt; ++k) {
					bmat[k + j * bmat_dim1] += sum * zmat[k + jj * zmat_dim1];
				}
			}
			for (long i = 1; i <= n; ++i) {
				const long ip = i + npt;
				temp = w[i];
				for (long j = 1; j <= i; ++j) {
					bmat[ip + j * bmat_dim1] += temp * w[j];
				}
			}
		}

		/*     The following instructions complete the shift, including the changes */
		/*     to the second derivative parameters of the quadratic model. */
		ih = 0;
		for (long j = 1; j <= n; ++j) {
			w[j] = -0.5 * sumpq * xopt[j];
			for (long k = 1; k <= npt; ++k) {
				w[j] += pq[k] * xpt[k + j * xpt_dim1];
				xpt[k + j * xpt_dim1] -= xopt[j];
			}
			for (long i = 1; i <= j; ++i) {
				++ih;
				hq[ih] = hq[ih] + w[i] * xopt[j] + xopt[i] * w[j];
				bmat[npt + i + j * bmat_dim1] = bmat[npt + j + i * bmat_dim1];
			}
		}
		for (long i = 1; i <= n; ++i) {
			xbase[i] += xopt[i];
			xnew[i] -= xopt[i];
			sl[i] -= xopt[i];
			su[i] -= xopt[i];
			xopt[i] = 0.;
		}
		xoptsq = 0.;
	}
	if (ntrits == 0) {
		goto L210;
	}
	goto L230;

	/*     XBASE is also moved to XOPT by a call of RESCUE. This calculation is */
	/*     more expensive than the previous shift, because new matrices BMAT and */
	/*     ZMAT are generated from scratch, which may include the replacement of */
	/*     interpolation points whose positions seem to be causing near linear */
	/*     dependence in the interpolation conditions. Therefore RESCUE is called */
	/*     only if rounding errors have reduced by at least a factor of two the */
	/*     denominator of the formula for updating the H matrix. It provides a */
	/*     useful safeguard, but is not invoked in most applications of BOBYQA. */
	L190: nfsav = nf;
	kbase = kopt;
	rescue(function, n, npt, xl, xu, maxfun, xbase, xpt + xpt_offset, fval,
			xopt, gopt, hq, pq, bmat + bmat_offset, zmat + zmat_offset, ndim,
			sl, su, nf, delta, kopt, vlag, w - 2, w + n + np - 1,
			w + ndim + np - 1, fev);

	/*     XOPT is updated now in case the branch below to label 720 is taken. */
	/*     Any updating of GOPT occurs after the branch below to label 20, which */
	/*     leads to a trust region iteration as does the branch to label 60. */
	xoptsq = 0.;
	if (kopt != kbase) {
		for (long i = 1; i <= n; ++i) {
			xopt[i] = xpt[kopt + i * xpt_dim1];
			xoptsq += xopt[i] * xopt[i];
		}
	}
	if (nf < 0) {
		nf = maxfun;
		goto L720;
	}
	nresc = nf;
	if (nfsav < nf) {
		nfsav = nf;
		goto L20;
	}
	if (ntrits > 0) {
		goto L60;
	}

	/*     Pick two alternative vectors of variables, relative to XBASE, that */
	/*     are suitable as new positions of the KNEW-th interpolation point. */
	/*     Firstly, XNEW is set to the point on a line through XOPT and another */
	/*     interpolation point that minimizes the predicted value of the next */
	/*     denominator, subject to ||XNEW - XOPT|| .LEQ. ADELT and to the SL */
	/*     and SU bounds. Secondly, XALT is set to the best feasible point on */
	/*     a constrained version of the Cauchy step of the KNEW-th Lagrange */
	/*     function, the corresponding value of the square of this function */
	/*     being returned in CAUCHY. The choice between these alternatives is */
	/*     going to be made when the denominator is calculated. */
	L210: altmov(n, npt, xpt + xpt_offset, xopt, bmat + bmat_offset,
			zmat + zmat_offset, ndim, sl, su, kopt, knew, adelt, xnew, xalt,
			alpha, cauchy, w, w + np - 1, w + ndim);
	for (long i = 1; i <= n; ++i) {
		d[i] = xnew[i] - xopt[i];
	}

	/*     Calculate VLAG and BETA for the current choice of D. The scalar */
	/*     product of D with XPT(K,.) is going to be held in W(NPT+K) for */
	/*     use when VQUAD is calculated. */
	L230: for (long k = 1; k <= npt; ++k) {
		suma = 0.;
		sumb = 0.;
		sum = 0.;
		for (long j = 1; j <= n; ++j) {
			suma += xpt[k + j * xpt_dim1] * d[j];
			sumb += xpt[k + j * xpt_dim1] * xopt[j];
			sum += bmat[k + j * bmat_dim1] * d[j];
		}
		w[k] = suma * (0.5 * suma + sumb);
		vlag[k] = sum;
		w[npt + k] = suma;
	}
	beta = 0.;
	for (long jj = 1; jj <= nptm; ++jj) {
		sum = 0.;
		for (long k = 1; k <= npt; ++k) {
			sum += zmat[k + jj * zmat_dim1] * w[k];
		}
		beta -= sum * sum;
		for (long k = 1; k <= npt; ++k) {
			vlag[k] += sum * zmat[k + jj * zmat_dim1];
		}
	}
	dsq = 0.;
	bsum = 0.;
	dx = 0.;
	for (long j = 1; j <= n; ++j) {
		dsq += d[j] * d[j];
		sum = 0.;
		for (long k = 1; k <= npt; ++k) {
			sum += w[k] * bmat[k + j * bmat_dim1];
		}
		bsum += sum * d[j];
		jp = npt + j;
		for (long i = 1; i <= n; ++i) {
			sum += bmat[jp + i * bmat_dim1] * d[i];
		}
		vlag[jp] = sum;
		bsum += sum * d[j];
		dx += d[j] * xopt[j];
	}
	beta = dx * dx + dsq * (xoptsq + dx + dx + 0.5 * dsq) + beta - bsum;
	vlag[kopt] += 1.;

	/*     If NTRITS is zero, the denominator may be increased by replacing */
	/*     the step D of ALTMOV by a Cauchy step. Then RESCUE may be called if */
	/*     rounding errors have damaged the chosen denominator. */
	if (ntrits == 0) {
		denom = vlag[knew] * vlag[knew] + alpha * beta;
		if (denom < cauchy && cauchy > 0.) {
			for (long i = 1; i <= n; ++i) {
				xnew[i] = xalt[i];
				d[i] = xnew[i] - xopt[i];
			}
			cauchy = 0.;
			goto L230;
		}
		if (denom <= 0.5 * vlag[knew] * vlag[knew]) {
			if (nf > nresc) {
				goto L190;
			}
			goto L720;
		}

		/*     Alternatively, if NTRITS is positive, then set KNEW to the index of */
		/*     the next interpolation point to be deleted to make room for a trust */
		/*     region step. Again RESCUE may be called if rounding errors have damaged */
		/*     the chosen denominator, which is the reason for attempting to select */
		/*     KNEW before calculating the next value of the objective function. */
	} else {
		delsq = delta * delta;
		scaden = 0.;
		biglsq = 0.;
		knew = 0;
		for (long k = 1; k <= npt; ++k) {
			if (k == kopt) {
				goto L350;
			}
			hdiag = 0.;
			for (long jj = 1; jj <= nptm; ++jj) {
				hdiag += std::pow(zmat[k + jj * zmat_dim1], 2);
			}
			den = beta * hdiag + vlag[k] * vlag[k];
			distsq = 0.0;
			for (long j = 1; j <= n; ++j) {
				distsq += std::pow(xpt[k + j * xpt_dim1] - xopt[j], 2);
			}
			temp = std::max(1., std::pow(distsq / delsq, 2));
			if (temp * den > scaden) {
				scaden = temp * den;
				knew = k;
				denom = den;
			}
			biglsq = std::max(biglsq, temp * vlag[k] * vlag[k]);

			L350: ;
		}
		if (scaden <= 0.5 * biglsq) {
			if (nf > nresc) {
				goto L190;
			}
			goto L720;
		}
	}

	/*     Put the variables for the next calculation of the objective function */
	/*       in XNEW, with any adjustments for the bounds. */
	/*     Calculate the value of the objective function at XBASE+XNEW, unless */
	/*       the limit on the number of calculations of F has been reached. */
	L360: for (long i = 1; i <= n; ++i) {
		x[i] = std::min(std::max(xl[i], xbase[i] + xnew[i]), xu[i]);
		if (xnew[i] == sl[i]) {
			x[i] = xl[i];
		}
		if (xnew[i] == su[i]) {
			x[i] = xu[i];
		}
	}
	if (nf >= maxfun) {
		goto L720;
	}
	++nf;
	f = function(x + 1);
	fev++;
	if (ntrits == -1) {
		fsave = f;
		goto L720;
	}

	/*     Use the quadratic model to predict the change in F due to the step D, */
	/*       and set DIFF to the error of this prediction. */
	fopt = fval[kopt];
	vquad = 0.;
	ih = 0;
	for (long j = 1; j <= n; ++j) {
		vquad += d[j] * gopt[j];
		for (long i = 1; i <= j; ++i) {
			++ih;
			temp = d[i] * d[j];
			if (i == j) {
				temp = 0.5 * temp;
			}
			vquad += hq[ih] * temp;
		}
	}
	for (long k = 1; k <= npt; ++k) {
		vquad += 0.5 * pq[k] * std::pow(w[npt + k], 2);
	}
	diff = f - fopt - vquad;
	diffc = diffb;
	diffb = diffa;
	diffa = std::abs(diff);
	if (dnorm > rho) {
		nfsav = nf;
	}

	/*     Pick the next value of DELTA after a trust region step. */
	if (ntrits > 0) {
		if (vquad >= 0.) {
			goto L720;
		}
		ratio = (f - fopt) / vquad;
		if (ratio <= 0.1) {
			delta = std::min(0.5 * delta, dnorm);
		} else if (ratio <= .7) {
			delta = std::max(0.5 * delta, dnorm);
		} else {
			delta = std::max(0.5 * delta, dnorm + dnorm);
		}
		if (delta <= rho * 1.5) {
			delta = rho;
		}

		/*     Recalculate KNEW and DENOM if the new F is less than FOPT. */
		if (f < fopt) {
			ksav = knew;
			densav = denom;
			delsq = delta * delta;
			scaden = 0.;
			biglsq = 0.;
			knew = 0;
			for (long k = 1; k <= npt; ++k) {
				hdiag = 0.;
				for (long jj = 1; jj <= nptm; ++jj) {
					hdiag += std::pow(zmat[k + jj * zmat_dim1], 2);
				}
				den = beta * hdiag + vlag[k] * vlag[k];
				distsq = 0.;
				for (long j = 1; j <= n; ++j) {
					distsq += std::pow(xpt[k + j * xpt_dim1] - xnew[j], 2);
				}
				temp = std::max(1., std::pow(distsq / delsq, 2));
				if (temp * den > scaden) {
					scaden = temp * den;
					knew = k;
					denom = den;
				}
				biglsq = std::max(biglsq, temp * vlag[k] * vlag[k]);
			}
			if (scaden <= 0.5 * biglsq) {
				knew = ksav;
				denom = densav;
			}
		}
	}

	/*     Update BMAT and ZMAT, so that the KNEW-th interpolation point can be */
	/*     moved. Also update the second derivative terms of the model. */
	update(n, npt, bmat + bmat_offset, zmat + zmat_offset, ndim, vlag, beta,
			denom, knew, w);
	ih = 0;
	pqold = pq[knew];
	pq[knew] = 0.;
	for (long i = 1; i <= n; ++i) {
		temp = pqold * xpt[knew + i * xpt_dim1];
		for (long j = 1; j <= i; ++j) {
			++ih;
			hq[ih] += temp * xpt[knew + j * xpt_dim1];
		}
	}
	for (long jj = 1; jj <= nptm; ++jj) {
		temp = diff * zmat[knew + jj * zmat_dim1];
		for (long k = 1; k <= npt; ++k) {
			pq[k] += temp * zmat[k + jj * zmat_dim1];
		}
	}

	/*     Include the new interpolation point, and make the changes to GOPT at */
	/*     the old XOPT that are caused by the updating of the quadratic model. */
	fval[knew] = f;
	for (long i = 1; i <= n; ++i) {
		xpt[knew + i * xpt_dim1] = xnew[i];
		w[i] = bmat[knew + i * bmat_dim1];
	}
	for (long k = 1; k <= npt; ++k) {
		suma = 0.;
		for (long jj = 1; jj <= nptm; ++jj) {
			suma += zmat[knew + jj * zmat_dim1] * zmat[k + jj * zmat_dim1];
		}
		sumb = 0.;
		for (long j = 1; j <= n; ++j) {
			sumb += xpt[k + j * xpt_dim1] * xopt[j];
		}
		temp = suma * sumb;
		for (long i = 1; i <= n; ++i) {
			w[i] += temp * xpt[k + i * xpt_dim1];
		}
	}
	for (long i = 1; i <= n; ++i) {
		gopt[i] += diff * w[i];
	}

	/*     Update XOPT, GOPT and KOPT if the new calculated F is less than FOPT. */
	if (f < fopt) {
		kopt = knew;
		xoptsq = 0.;
		ih = 0;
		for (long j = 1; j <= n; ++j) {
			xopt[j] = xnew[j];
			xoptsq += xopt[j] * xopt[j];
			for (long i = 1; i <= j; ++i) {
				++ih;
				if (i < j) {
					gopt[j] += hq[ih] * d[i];
				}
				gopt[i] += hq[ih] * d[j];
			}
		}
		for (long k = 1; k <= npt; ++k) {
			temp = 0.;
			for (long j = 1; j <= n; ++j) {
				temp += xpt[k + j * xpt_dim1] * d[j];
			}
			temp = pq[k] * temp;
			for (long i = 1; i <= n; ++i) {
				gopt[i] += temp * xpt[k + i * xpt_dim1];
			}
		}
	}

	/*     Calculate the parameters of the least Frobenius norm interpolant to */
	/*     the current data, the gradient of this interpolant at XOPT being put */
	/*     into VLAG(NPT+I), I=1,2,...,N. */
	if (ntrits > 0) {
		for (long k = 1; k <= npt; ++k) {
			vlag[k] = fval[k] - fval[kopt];
			w[k] = 0.;
		}
		for (long j = 1; j <= nptm; ++j) {
			sum = 0.;
			for (long k = 1; k <= npt; ++k) {
				sum += zmat[k + j * zmat_dim1] * vlag[k];
			}
			for (long k = 1; k <= npt; ++k) {
				w[k] += sum * zmat[k + j * zmat_dim1];
			}
		}
		for (long k = 1; k <= npt; ++k) {
			sum = 0.;
			for (long j = 1; j <= n; ++j) {
				sum += xpt[k + j * xpt_dim1] * xopt[j];
			}
			w[k + npt] = w[k];
			w[k] = sum * w[k];
		}
		gqsq = 0.;
		gisq = 0.;
		for (long i = 1; i <= n; ++i) {
			sum = 0.;
			for (long k = 1; k <= npt; ++k) {
				sum = sum + bmat[k + i * bmat_dim1] * vlag[k]
						+ xpt[k + i * xpt_dim1] * w[k];
			}
			if (xopt[i] == sl[i]) {
				gqsq += std::pow(std::min(0., gopt[i]), 2);
				gisq += std::pow(std::min(0., sum), 2);
			} else if (xopt[i] == su[i]) {
				gqsq += std::pow(std::max(0., gopt[i]), 2);
				gisq += std::pow(std::max(0., sum), 2);
			} else {
				gqsq += gopt[i] * gopt[i];
				gisq += sum * sum;
			}
			vlag[npt + i] = sum;
		}

		/*     Test whether to replace the new quadratic model by the least Frobenius */
		/*     norm interpolant, making the replacement if the test is satisfied. */
		++itest;
		if (gqsq < 10. * gisq) {
			itest = 0;
		}
		if (itest >= 3) {
			const long i_n = std::max(npt, nh);
			for (long i = 1; i <= i_n; ++i) {
				if (i <= n) {
					gopt[i] = vlag[npt + i];
				}
				if (i <= npt) {
					pq[i] = w[npt + i];
				}
				if (i <= nh) {
					hq[i] = 0.;
				}
				itest = 0;
			}
		}
	}

	/*     If a trust region step has provided a sufficient decrease in F, then */
	/*     branch for another trust region calculation. The case NTRITS=0 occurs */
	/*     when the new interpolation point was reached by an alternative step. */
	if (ntrits == 0) {
		goto L60;
	}
	if (f <= fopt + 0.1 * vquad) {
		goto L60;
	}

	/*     Alternatively, find out if the interpolation points are close enough */
	/*       to the best point so far. */
	distsq = std::max(std::pow(2. * delta, 2), std::pow(10. * rho, 2));
	L650: knew = 0;
	for (long k = 1; k <= npt; ++k) {
		sum = 0.;
		for (long j = 1; j <= n; ++j) {
			sum += std::pow(xpt[k + j * xpt_dim1] - xopt[j], 2);
		}
		if (sum > distsq) {
			knew = k;
			distsq = sum;
		}
	}

	/*     If KNEW is positive, then ALTMOV finds alternative new positions for */
	/*     the KNEW-th interpolation point within distance ADELT of XOPT. It is */
	/*     reached via label 90. Otherwise, there is a branch to label 60 for */
	/*     another trust region iteration, unless the calculations with the */
	/*     current RHO are complete. */
	if (knew > 0) {
		dist = std::sqrt(distsq);
		if (ntrits == -1) {
			delta = std::min(0.1 * delta, 0.5 * dist);
			if (delta <= rho * 1.5) {
				delta = rho;
			}
		}
		ntrits = 0;
		adelt = std::max(std::min(0.1 * dist, delta), rho);
		dsq = adelt * adelt;
		goto L90;
	}
	if (ntrits == -1) {
		goto L680;
	}
	if (ratio > 0.) {
		goto L60;
	}
	if (std::max(delta, dnorm) > rho) {
		goto L60;
	}

	/*     The calculations with the current value of RHO are complete. Pick the */
	/*       next values of RHO and DELTA. */
	L680: if (rho > rhoend) {
		delta = 0.5 * rho;
		ratio = rho / rhoend;
		if (ratio <= 16.) {
			rho = rhoend;
		} else if (ratio <= 250.) {
			rho = std::sqrt(ratio) * rhoend;
		} else {
			rho = 0.1 * rho;
		}
		delta = std::max(delta, rho);
		ntrits = 0;
		nfsav = nf;
		goto L60;
	}

	/*     Return from the calculation, after another Newton-Raphson step, if */
	/*       it is too short to have been tried before. */
	if (ntrits == -1) {
		goto L360;
	}
	L720: if (fval[kopt] <= fsave) {
		for (long i = 1; i <= n; ++i) {
			x[i] = std::min(std::max(xl[i], xbase[i] + xopt[i]), xu[i]);
			if (xopt[i] == sl[i]) {
				x[i] = xl[i];
			}
			if (xopt[i] == su[i]) {
				x[i] = xu[i];
			}
		}
		f = fval[kopt];
	}
	return f;
}

double Bobyqa::bobyqa(const multivariate &function, const long n,
		const long npt, double *x, const double *xl, const double *xu,
		const double rhobeg, const double rhoend, const long maxfun, double *w,
		long &fev) {

	/*     This subroutine seeks the least value of a function of many variables, */
	/*     by applying a trust region method that forms quadratic models by */
	/*     interpolation. There is usually some freedom in the interpolation */
	/*     conditions, which is taken up by minimizing the Frobenius norm of */
	/*     the change to the second derivative of the model, beginning with the */
	/*     zero matrix. The values of the variables are constrained by upper and */
	/*     lower bounds. The arguments of the subroutine are as follows. */

	/*     N must be set to the number of variables and must be at least two. */
	/*     NPT is the number of interpolation conditions. Its value must be in */
	/*       the interval [N+2,(N+1)(N+2)/2]. Choices that exceed 2*N+1 are not */
	/*       recommended. */
	/*     Initial values of the variables must be set in X(1),X(2),...,X(N). They */
	/*       will be changed to the values that give the least calculated F. */
	/*     For I=1,2,...,N, XL(I) and XU(I) must provide the lower and upper */
	/*       bounds, respectively, on X(I). The construction of quadratic models */
	/*       requires XL(I) to be strictly less than XU(I) for each I. Further, */
	/*       the contribution to a model from changes to the I-th variable is */
	/*       damaged severely by rounding errors if XU(I)-XL(I) is too small. */
	/*     RHOBEG and RHOEND must be set to the initial and final values of a trust */
	/*       region radius, so both must be positive with RHOEND no greater than */
	/*       RHOBEG. Typically, RHOBEG should be about one tenth of the greatest */
	/*       expected change to a variable, while RHOEND should indicate the */
	/*       accuracy that is required in the final values of the variables. An */
	/*       error return occurs if any of the differences XU(I)-XL(I), I=1,...,N, */
	/*       is less than 2*RHOBEG. */
	/*     MAXFUN must be set to an upper bound on the number of calls of CALFUN. */
	/*     The array W will be used for working space. Its length must be at least */
	/*       (NPT+5)*(NPT+N)+3*N*(N+5)/2. */

	/* Parameter adjustments */
	--w;
	--xu;
	--xl;
	--x;

	/* Function Body */
	const long np = n + 1;

	/*     Return if the value of NPT is unacceptable. */
	if (npt < n + 2 || npt > (n + 2) * np / 2) {
		return 0.;
	}

	/*     Partition the working space array, so that different parts of it can */
	/*     be treated separately during the calculation of BOBYQB. The partition */
	/*     requires the first (NPT+2)*(NPT+N)+3*N*(N+5)/2 elements of W plus the */
	/*     space that is taken by the last array in the argument list of BOBYQB. */
	const long ndim = npt + n;
	const long ixp = 1 + n;
	const long ifv = ixp + n * npt;
	const long ixo = ifv + npt;
	const long igo = ixo + n;
	const long ihq = igo + n;
	const long ipq = ihq + n * np / 2;
	const long ibmat = ipq + npt;
	const long izmat = ibmat + ndim * n;
	const long isl = izmat + npt * (npt - np);
	const long isu = isl + n;
	const long ixn = isu + n;
	const long ixa = ixn + n;
	const long id_ = ixa + n;
	const long ivl = id_ + n;
	const long iw = ivl + ndim;

	/*     Return if there is insufficient space between the bounds. Modify the */
	/*     initial X if necessary in order to avoid conflicts between the bounds */
	/*     and the construction of the first quadratic model. The lower and upper */
	/*     bounds on moves from the updated X are set now, in the ISL and ISU */
	/*     partitions of W, in order to provide useful and exact information about */
	/*     components of X that become within distance RHOBEG from their bounds. */
	for (long j = 1; j <= n; ++j) {
		const double temp = xu[j] - xl[j];
		if (temp < rhobeg + rhobeg) {
			return 0.;
		}
		const long jsl = isl + j - 1;
		const long jsu = jsl + n;
		w[jsl] = xl[j] - x[j];
		w[jsu] = xu[j] - x[j];
		if (w[jsl] >= -(rhobeg)) {
			if (w[jsl] >= 0.) {
				x[j] = xl[j];
				w[jsl] = 0.;
				w[jsu] = temp;
			} else {
				x[j] = xl[j] + rhobeg;
				w[jsl] = -(rhobeg);
				w[jsu] = std::max(xu[j] - x[j], rhobeg);
			}
		} else if (w[jsu] <= rhobeg) {
			if (w[jsu] <= 0.) {
				x[j] = xu[j];
				w[jsl] = -temp;
				w[jsu] = 0.;
			} else {
				x[j] = xu[j] - rhobeg;
				w[jsl] = std::min(xl[j] - x[j], -rhobeg);
				w[jsu] = rhobeg;
			}
		}
	}

	/*     Make the call of BOBYQB. */
	return bobyqb(function, n, npt, x, xl, xu, rhobeg, rhoend, maxfun, w,
			w + ixp, w + ifv - 1, w + ixo - 1, w + igo - 1, w + ihq - 1,
			w + ipq - 1, w + ibmat, w + izmat, ndim, w + isl - 1, w + isu - 1,
			w + ixn - 1, w + ixa - 1, w + id_ - 1, w + ivl - 1, w + iw - 1, fev);
}
