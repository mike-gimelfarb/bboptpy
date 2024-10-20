/* The MIT License
 Copyright (c) 2004, by M.J.D. Powell <mjdp@cam.ac.uk>
 2008, by Attractive Chaos <attractivechaos@aol.co.uk>

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include "../../random.hpp"

#include "newuoa.h"

using Random = effolkronium::random_static;

Newuoa::Newuoa(int mfev, int np, double rho, double tol) {
	_np = np;
	_rho = rho;
	_tol = tol;
	_mfev = mfev;
}

void Newuoa::init(const multivariate_problem &f, const double *guess) {
	// nothing to do here
}

void Newuoa::iterate() {
	// nothing to do here
}

multivariate_solution Newuoa::solution(){
	return {{}, 0, false};
}

multivariate_solution Newuoa::optimize(const multivariate_problem &f,
		const double *guess) {
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [NEWUOA]: problem constraints will be ignored."
				<< std::endl;
	}

	// initialize point
	std::vector<double> x(guess, guess + f._n);

	// adjust population size
	int npt = _np;
	if (npt < f._n + 2) {
		npt = f._n + 2;
		std::cerr << "Warning [NEWUOA]: population size too small - adjusted."
				<< std::endl;
	} else if (npt > (f._n + 2) * (f._n + 1) / 2) {
		npt = (f._n + 2) * (f._n + 1) / 2;
		std::cerr << "Warning [NEWUOA]: population size too large - adjusted."
				<< std::endl;
	}

	// adjust rho (should be about 1/5 of the problem size)
	if (_rho <= 0.) {
		for (int i = 0; i < f._n; i++) {
			_rho = std::max(_rho, (f._upper[i] - f._lower[i]) / 5.);
		}
		std::cerr << "Warning [NEWUOA]: rho is inadequate - adjusted."
				<< std::endl;
	}

	// create work array
	std::vector<double> w(
			(npt + 13) * (npt + f._n) + 3 * f._n * (f._n + 3) / 2);

	// main call
	int fev = 0;
	newuoa(f._f, f._n, npt, &x[0], _rho, _tol, _mfev, &w[0], fev);
	return {x, fev, false};
}

void Newuoa::biglag(long n, long npt, double *xopt, double *xpt, double *bmat,
		double *zmat, long *idz, long *ndim, long *knew, double *delta,
		double *d, double *alpha, double *hcol, double *gc, double *gd,
		double *s, double *w) {

	long xpt_dim1, xpt_offset, bmat_dim1, bmat_offset, zmat_dim1, zmat_offset,
			i1, i2, ii, j, k, iu, nptm, iterc, isave;
	double sp, ss, cf1, cf2, cf3, cf4, cf5, dhd, cth, tau, sth, sum, temp, step,
			angle, scale, denom, delsq, tempa, tempb, twopi, taubeg, tauold,
			taumax, d1, dd, gg;

	/* Parameter adjustments */
	tempa = tempb = 0.;
	zmat_dim1 = npt;
	zmat_offset = 1 + zmat_dim1;
	zmat -= zmat_offset;
	xpt_dim1 = npt;
	xpt_offset = 1 + xpt_dim1;
	xpt -= xpt_offset;
	--xopt;
	bmat_dim1 = *ndim;
	bmat_offset = 1 + bmat_dim1;
	bmat -= bmat_offset;
	--d;
	--hcol;
	--gc;
	--gd;
	--s;
	--w;

	/* Functiontion Body */
	twopi = 2. * M_PI;
	delsq = *delta * *delta;
	nptm = npt - n - 1;

	/* Set the first NPT components of HCOBLL to the leading elements of
	 * the KNEW-th column of H. */
	i1 = npt;
	for (k = 1; k <= i1; ++k) {
		hcol[k] = 0;
	}
	i1 = nptm;
	for (j = 1; j <= i1; ++j) {
		temp = zmat[*knew + j * zmat_dim1];
		if (j < *idz) {
			temp = -temp;
		}
		i2 = npt;
		for (k = 1; k <= i2; ++k) {
			hcol[k] += temp * zmat[k + j * zmat_dim1];
		}
	}
	*alpha = hcol[*knew];

	/* Set the unscaled initial direction D. Form the gradient of BLLFUNC
	 * atXOPT, and multiply D by the second derivative matrix of
	 * BLLFUNC. */
	dd = 0;
	i2 = n;
	for (ii = 1; ii <= i2; ++ii) {
		d[ii] = xpt[*knew + ii * xpt_dim1] - xopt[ii];
		gc[ii] = bmat[*knew + ii * bmat_dim1];
		gd[ii] = 0;

		/* Computing 2nd power */
		d1 = d[ii];
		dd += d1 * d1;
	}
	i2 = npt;
	for (k = 1; k <= i2; ++k) {
		temp = 0;
		sum = 0;
		i1 = n;
		for (j = 1; j <= i1; ++j) {
			temp += xpt[k + j * xpt_dim1] * xopt[j];
			sum += xpt[k + j * xpt_dim1] * d[j];
		}
		temp = hcol[k] * temp;
		sum = hcol[k] * sum;
		i1 = n;
		for (ii = 1; ii <= i1; ++ii) {
			gc[ii] += temp * xpt[k + ii * xpt_dim1];
			gd[ii] += sum * xpt[k + ii * xpt_dim1];
		}
	}

	/* Scale D and GD, with a sign change if required. Set S to another
	 * vector in the initial two dimensional subspace. */
	gg = sp = dhd = 0;
	i1 = n;
	for (ii = 1; ii <= i1; ++ii) {

		/* Computing 2nd power */
		d1 = gc[ii];
		gg += d1 * d1;
		sp += d[ii] * gc[ii];
		dhd += d[ii] * gd[ii];
	}
	scale = *delta / std::sqrt(dd);
	if (sp * dhd < 0) {
		scale = -scale;
	}
	temp = 0;
	if (sp * sp > dd * .99 * gg) {
		temp = 1.;
	}
	tau = scale * (std::fabs(sp) + 0.5 * scale * std::fabs(dhd));
	if (gg * delsq < tau * .01 * tau) {
		temp = 1.;
	}
	i1 = n;
	for (ii = 1; ii <= i1; ++ii) {
		d[ii] = scale * d[ii];
		gd[ii] = scale * gd[ii];
		s[ii] = gc[ii] + temp * gd[ii];
	}

	/* Begin the iteration by overwriting S with a vector that has the
	 * required length and direction, except that termination occurs if
	 * the given D and S are nearly parallel. */
	for (iterc = 0; iterc != n; ++iterc) {
		dd = sp = ss = 0;
		i1 = n;
		for (ii = 1; ii <= i1; ++ii) {

			/* Computing 2nd power */
			d1 = d[ii];
			dd += d1 * d1;
			sp += d[ii] * s[ii];

			/* Computing 2nd power */
			d1 = s[ii];
			ss += d1 * d1;
		}
		temp = dd * ss - sp * sp;
		if (temp <= dd * 1e-8 * ss) {
			return;
		}
		denom = std::sqrt(temp);
		i1 = n;
		for (ii = 1; ii <= i1; ++ii) {
			s[ii] = (dd * s[ii] - sp * d[ii]) / denom;
			w[ii] = 0;
		}

		/* Calculate the coefficients of the objective function on the
		 * circle, beginning with the multiplication of S by the second
		 * derivative matrix. */
		i1 = npt;
		for (k = 1; k <= i1; ++k) {
			sum = 0;
			i2 = n;
			for (j = 1; j <= i2; ++j) {
				sum += xpt[k + j * xpt_dim1] * s[j];
			}
			sum = hcol[k] * sum;
			i2 = n;
			for (ii = 1; ii <= i2; ++ii) {
				w[ii] += sum * xpt[k + ii * xpt_dim1];
			}
		}
		cf1 = cf2 = cf3 = cf4 = cf5 = 0;
		i2 = n;
		for (ii = 1; ii <= i2; ++ii) {
			cf1 += s[ii] * w[ii];
			cf2 += d[ii] * gc[ii];
			cf3 += s[ii] * gc[ii];
			cf4 += d[ii] * gd[ii];
			cf5 += s[ii] * gd[ii];
		}
		cf1 = 0.5 * cf1;
		cf4 = 0.5 * cf4 - cf1;

		/* Seek the value of the angle that maximizes the modulus of TAU. */
		taubeg = cf1 + cf2 + cf4;
		taumax = tauold = taubeg;
		isave = 0;
		iu = 49;
		temp = twopi / (iu + 1.);
		i2 = iu;
		for (ii = 1; ii <= i2; ++ii) {
			angle = 1. * ii * temp;
			cth = std::cos(angle);
			sth = std::sin(angle);
			tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth;
			if (std::fabs(tau) > std::fabs(taumax)) {
				taumax = tau;
				isave = ii;
				tempa = tauold;
			} else if (ii == isave + 1) {
				tempb = tau;
			}
			tauold = tau;
		}
		if (isave == 0) {
			tempa = tau;
		}
		if (isave == iu) {
			tempb = taubeg;
		}
		step = 0;
		if (tempa != tempb) {
			tempa -= taumax;
			tempb -= taumax;
			step = 0.5 * (tempa - tempb) / (tempa + tempb);
		}
		angle = temp * (1. * isave + step);

		/* Calculate the new D and GD. Then test for convergence. */
		cth = std::cos(angle);
		sth = std::sin(angle);
		tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth;
		i2 = n;
		for (ii = 1; ii <= i2; ++ii) {
			d[ii] = cth * d[ii] + sth * s[ii];
			gd[ii] = cth * gd[ii] + sth * w[ii];
			s[ii] = gc[ii] + gd[ii];
		}
		if (std::fabs(tau) <= std::fabs(taubeg) * 1.1) {
			return;
		}
	}
}

void Newuoa::bigden(long n, long npt, double *xopt, double *xpt, double *bmat,
		double *zmat, long *idz, long *ndim, long *kopt, long *knew, double *d,
		double *w, double *vlag, double *beta, double *s, double *wvec,
		double *prod) {

	long xpt_dim1, xpt_offset, bmat_dim1, bmat_offset, zmat_dim1, zmat_offset,
			wvec_dim1, wvec_offset, prod_dim1, prod_offset, i1, i2, i, j, k,
			isave, iterc, jc, ip, iu, nw, ksav, nptm;
	double dd, d1, ds, ss, den[9], par[9], tau, sum, diff, temp, step, alpha,
			angle, denex[9], tempa, tempb, tempc, ssden, dtest, xoptd, twopi,
			xopts, denold, denmax, densav, dstemp, sumold, sstemp, xoptsq;

	/* Parameter adjustments */
	zmat_dim1 = npt;
	zmat_offset = 1 + zmat_dim1;
	zmat -= zmat_offset;
	xpt_dim1 = npt;
	xpt_offset = 1 + xpt_dim1;
	xpt -= xpt_offset;
	--xopt;
	prod_dim1 = *ndim;
	prod_offset = 1 + prod_dim1;
	prod -= prod_offset;
	wvec_dim1 = *ndim;
	wvec_offset = 1 + wvec_dim1;
	wvec -= wvec_offset;
	bmat_dim1 = *ndim;
	bmat_offset = 1 + bmat_dim1;
	bmat -= bmat_offset;
	--d;
	--w;
	--vlag;
	--s;

	/* Functiontion Body */
	twopi = std::atan(1.) * 8.;
	nptm = npt - n - 1;

	/* Store the first NPT elements of the KNEW-th column of H in W(N+1)
	 * to W(N+NPT). */
	i1 = npt;
	for (k = 1; k <= i1; ++k) {
		w[n + k] = 0;
	}
	i1 = nptm;
	for (j = 1; j <= i1; ++j) {
		temp = zmat[*knew + j * zmat_dim1];
		if (j < *idz) {
			temp = -temp;
		}
		i2 = npt;
		for (k = 1; k <= i2; ++k) {
			w[n + k] += temp * zmat[k + j * zmat_dim1];
		}
	}
	alpha = w[n + *knew];

	/* The initial search direction D is taken from the last call of
	 * BIGBDLAG, and the initial S is set below, usually to the direction
	 * from X_OPT to X_KNEW, but a different direction to an
	 * interpolation point may be chosen, in order to prevent S from
	 * being nearly parallel to D. */
	dd = ds = ss = xoptsq = 0;
	i2 = n;
	for (i = 1; i <= i2; ++i) {

		/* Computing 2nd power */
		d1 = d[i];
		dd += d1 * d1;
		s[i] = xpt[*knew + i * xpt_dim1] - xopt[i];
		ds += d[i] * s[i];

		/* Computing 2nd power */
		d1 = s[i];
		ss += d1 * d1;

		/* Computing 2nd power */
		d1 = xopt[i];
		xoptsq += d1 * d1;
	}
	if (ds * ds > dd * .99 * ss) {
		ksav = *knew;
		dtest = ds * ds / ss;
		i2 = npt;
		for (k = 1; k <= i2; ++k) {
			if (k != *kopt) {
				dstemp = 0;
				sstemp = 0;
				i1 = n;
				for (i = 1; i <= i1; ++i) {
					diff = xpt[k + i * xpt_dim1] - xopt[i];
					dstemp += d[i] * diff;
					sstemp += diff * diff;
				}
				if (dstemp * dstemp / sstemp < dtest) {
					ksav = k;
					dtest = dstemp * dstemp / sstemp;
					ds = dstemp;
					ss = sstemp;
				}
			}
		}
		i2 = n;
		for (i = 1; i <= i2; ++i) {
			s[i] = xpt[ksav + i * xpt_dim1] - xopt[i];
		}
	}
	ssden = dd * ss - ds * ds;
	iterc = 0;
	densav = 0;

	/* Begin the iteration by overwriting S with a vector that has the
	 * required length and direction. */
	BDL70: ++iterc;
	temp = 1. / std::sqrt(ssden);
	xoptd = xopts = 0;
	i2 = n;
	for (i = 1; i <= i2; ++i) {
		s[i] = temp * (dd * s[i] - ds * d[i]);
		xoptd += xopt[i] * d[i];
		xopts += xopt[i] * s[i];
	}

	/* Set the coefficients of the first 2.0 terms of BETA. */
	tempa = 0.5 * xoptd * xoptd;
	tempb = 0.5 * xopts * xopts;
	den[0] = dd * (xoptsq + 0.5 * dd) + tempa + tempb;
	den[1] = 2. * xoptd * dd;
	den[2] = 2. * xopts * dd;
	den[3] = tempa - tempb;
	den[4] = xoptd * xopts;
	for (i = 6; i <= 9; ++i) {
		den[i - 1] = 0;
	}

	/* Put the coefficients of Wcheck in WVEC. */
	i2 = npt;
	for (k = 1; k <= i2; ++k) {
		tempa = tempb = tempc = 0;
		i1 = n;
		for (i = 1; i <= i1; ++i) {
			tempa += xpt[k + i * xpt_dim1] * d[i];
			tempb += xpt[k + i * xpt_dim1] * s[i];
			tempc += xpt[k + i * xpt_dim1] * xopt[i];
		}
		wvec[k + wvec_dim1] = 0.25 * (tempa * tempa + tempb * tempb);
		wvec[k + (wvec_dim1 << 1)] = tempa * tempc;
		wvec[k + wvec_dim1 * 3] = tempb * tempc;
		wvec[k + (wvec_dim1 << 2)] = 0.25 * (tempa * tempa - tempb * tempb);
		wvec[k + wvec_dim1 * 5] = 0.5 * tempa * tempb;
	}
	i2 = n;
	for (i = 1; i <= i2; ++i) {
		ip = i + npt;
		wvec[ip + wvec_dim1] = 0;
		wvec[ip + (wvec_dim1 << 1)] = d[i];
		wvec[ip + wvec_dim1 * 3] = s[i];
		wvec[ip + (wvec_dim1 << 2)] = 0;
		wvec[ip + wvec_dim1 * 5] = 0;
	}

	/* Put the coefficents of THETA*Wcheck in PROD. */
	for (jc = 1; jc <= 5; ++jc) {
		nw = npt;
		if (jc == 2 || jc == 3) {
			nw = *ndim;
		}
		i2 = npt;
		for (k = 1; k <= i2; ++k) {
			prod[k + jc * prod_dim1] = 0;
		}
		i2 = nptm;
		for (j = 1; j <= i2; ++j) {
			sum = 0;
			i1 = npt;
			for (k = 1; k <= i1; ++k) {
				sum += zmat[k + j * zmat_dim1] * wvec[k + jc * wvec_dim1];
			}
			if (j < *idz) {
				sum = -sum;
			}
			i1 = npt;
			for (k = 1; k <= i1; ++k) {
				prod[k + jc * prod_dim1] += sum * zmat[k + j * zmat_dim1];
			}
		}
		if (nw == *ndim) {
			i1 = npt;
			for (k = 1; k <= i1; ++k) {
				sum = 0;
				i2 = n;
				for (j = 1; j <= i2; ++j) {
					sum += bmat[k + j * bmat_dim1]
							* wvec[npt + j + jc * wvec_dim1];
				}
				prod[k + jc * prod_dim1] += sum;
			}
		}
		i1 = n;
		for (j = 1; j <= i1; ++j) {
			sum = 0;
			i2 = nw;
			for (i = 1; i <= i2; ++i) {
				sum += bmat[i + j * bmat_dim1] * wvec[i + jc * wvec_dim1];
			}
			prod[npt + j + jc * prod_dim1] = sum;
		}
	}

	/* Include in DEN the part of BETA that depends on THETA. */
	i1 = *ndim;
	for (k = 1; k <= i1; ++k) {
		sum = 0;
		for (i = 1; i <= 5; ++i) {
			par[i - 1] = 0.5 * prod[k + i * prod_dim1]
					* wvec[k + i * wvec_dim1];
			sum += par[i - 1];
		}
		den[0] = den[0] - par[0] - sum;
		tempa = prod[k + prod_dim1] * wvec[k + (wvec_dim1 << 1)]
				+ prod[k + (prod_dim1 << 1)] * wvec[k + wvec_dim1];
		tempb = prod[k + (prod_dim1 << 1)] * wvec[k + (wvec_dim1 << 2)]
				+ prod[k + (prod_dim1 << 2)] * wvec[k + (wvec_dim1 << 1)];
		tempc = prod[k + prod_dim1 * 3] * wvec[k + wvec_dim1 * 5]
				+ prod[k + prod_dim1 * 5] * wvec[k + wvec_dim1 * 3];
		den[1] = den[1] - tempa - 0.5 * (tempb + tempc);
		den[5] -= 0.5 * (tempb - tempc);
		tempa = prod[k + prod_dim1] * wvec[k + wvec_dim1 * 3]
				+ prod[k + prod_dim1 * 3] * wvec[k + wvec_dim1];
		tempb = prod[k + (prod_dim1 << 1)] * wvec[k + wvec_dim1 * 5]
				+ prod[k + prod_dim1 * 5] * wvec[k + (wvec_dim1 << 1)];
		tempc = prod[k + prod_dim1 * 3] * wvec[k + (wvec_dim1 << 2)]
				+ prod[k + (prod_dim1 << 2)] * wvec[k + wvec_dim1 * 3];
		den[2] = den[2] - tempa - 0.5 * (tempb - tempc);
		den[6] -= 0.5 * (tempb + tempc);
		tempa = prod[k + prod_dim1] * wvec[k + (wvec_dim1 << 2)]
				+ prod[k + (prod_dim1 << 2)] * wvec[k + wvec_dim1];
		den[3] = den[3] - tempa - par[1] + par[2];
		tempa = prod[k + prod_dim1] * wvec[k + wvec_dim1 * 5]
				+ prod[k + prod_dim1 * 5] * wvec[k + wvec_dim1];
		tempb = prod[k + (prod_dim1 << 1)] * wvec[k + wvec_dim1 * 3]
				+ prod[k + prod_dim1 * 3] * wvec[k + (wvec_dim1 << 1)];
		den[4] = den[4] - tempa - 0.5 * tempb;
		den[7] = den[7] - par[3] + par[4];
		tempa = prod[k + (prod_dim1 << 2)] * wvec[k + wvec_dim1 * 5]
				+ prod[k + prod_dim1 * 5] * wvec[k + (wvec_dim1 << 2)];
		den[8] -= 0.5 * tempa;
	}

	/* Extend DEN so that it holds all the coefficients of DENOM. */
	sum = 0;
	for (i = 1; i <= 5; ++i) {

		/* Computing 2nd power */
		d1 = prod[*knew + i * prod_dim1];
		par[i - 1] = 0.5 * (d1 * d1);
		sum += par[i - 1];
	}
	denex[0] = alpha * den[0] + par[0] + sum;
	tempa = 2.0 * prod[*knew + prod_dim1] * prod[*knew + (prod_dim1 << 1)];
	tempb = prod[*knew + (prod_dim1 << 1)] * prod[*knew + (prod_dim1 << 2)];
	tempc = prod[*knew + prod_dim1 * 3] * prod[*knew + prod_dim1 * 5];
	denex[1] = alpha * den[1] + tempa + tempb + tempc;
	denex[5] = alpha * den[5] + tempb - tempc;
	tempa = 2.0 * prod[*knew + prod_dim1] * prod[*knew + prod_dim1 * 3];
	tempb = prod[*knew + (prod_dim1 << 1)] * prod[*knew + prod_dim1 * 5];
	tempc = prod[*knew + prod_dim1 * 3] * prod[*knew + (prod_dim1 << 2)];
	denex[2] = alpha * den[2] + tempa + tempb - tempc;
	denex[6] = alpha * den[6] + tempb + tempc;
	tempa = 2.0 * prod[*knew + prod_dim1] * prod[*knew + (prod_dim1 << 2)];
	denex[3] = alpha * den[3] + tempa + par[1] - par[2];
	tempa = 2.0 * prod[*knew + prod_dim1] * prod[*knew + prod_dim1 * 5];
	denex[4] = alpha * den[4] + tempa
			+ prod[*knew + (prod_dim1 << 1)] * prod[*knew + prod_dim1 * 3];
	denex[7] = alpha * den[7] + par[3] - par[4];
	denex[8] = alpha * den[8]
			+ prod[*knew + (prod_dim1 << 2)] * prod[*knew + prod_dim1 * 5];

	/* Seek the value of the angle that maximizes the modulus of DENOM. */
	sum = denex[0] + denex[1] + denex[3] + denex[5] + denex[7];
	denold = denmax = sum;
	isave = 0;
	iu = 49;
	temp = twopi / (iu + 1.);
	par[0] = 1.;
	i1 = iu;
	for (i = 1; i <= i1; ++i) {
		angle = 1. * i * temp;
		par[1] = std::cos(angle);
		par[2] = std::sin(angle);
		for (j = 4; j <= 8; j += 2) {
			par[j - 1] = par[1] * par[j - 3] - par[2] * par[j - 2];
			par[j] = par[1] * par[j - 2] + par[2] * par[j - 3];
		}
		sumold = sum;
		sum = 0;
		for (j = 1; j <= 9; ++j) {
			sum += denex[j - 1] * par[j - 1];
		}
		if (std::fabs(sum) > std::fabs(denmax)) {
			denmax = sum;
			isave = i;
			tempa = sumold;
		} else if (i == isave + 1) {
			tempb = sum;
		}
	}
	if (isave == 0) {
		tempa = sum;
	}
	if (isave == iu) {
		tempb = denold;
	}
	step = 0;
	if (tempa != tempb) {
		tempa -= denmax;
		tempb -= denmax;
		step = 0.5 * (tempa - tempb) / (tempa + tempb);
	}
	angle = temp * (1. * isave + step);

	/* Calculate the new parameters of the denominator, the new VBDLAG
	 * vector and the new D. Then test for convergence. */
	par[1] = std::cos(angle);
	par[2] = std::sin(angle);
	for (j = 4; j <= 8; j += 2) {
		par[j - 1] = par[1] * par[j - 3] - par[2] * par[j - 2];
		par[j] = par[1] * par[j - 2] + par[2] * par[j - 3];
	}
	*beta = 0;
	denmax = 0;
	for (j = 1; j <= 9; ++j) {
		*beta += den[j - 1] * par[j - 1];
		denmax += denex[j - 1] * par[j - 1];
	}
	i1 = *ndim;
	for (k = 1; k <= i1; ++k) {
		vlag[k] = 0;
		for (j = 1; j <= 5; ++j) {
			vlag[k] += prod[k + j * prod_dim1] * par[j - 1];
		}
	}
	tau = vlag[*knew];
	dd = tempa = tempb = 0;
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		d[i] = par[1] * d[i] + par[2] * s[i];
		w[i] = xopt[i] + d[i];

		/* Computing 2nd power */
		d1 = d[i];
		dd += d1 * d1;
		tempa += d[i] * w[i];
		tempb += w[i] * w[i];
	}
	if (iterc >= n) {
		goto BDL340;
	}
	if (iterc > 1) {
		densav = std::fmax(densav, denold);
	}
	if (std::fabs(denmax) <= std::fabs(densav) * 1.1) {
		goto BDL340;
	}
	densav = denmax;

	/* Set S to 0.5 the gradient of the denominator with respect to
	 * D. Then branch for the next iteration. */
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		temp = tempa * xopt[i] + tempb * d[i] - vlag[npt + i];
		s[i] = tau * bmat[*knew + i * bmat_dim1] + alpha * temp;
	}
	i1 = npt;
	for (k = 1; k <= i1; ++k) {
		sum = 0;
		i2 = n;
		for (j = 1; j <= i2; ++j) {
			sum += xpt[k + j * xpt_dim1] * w[j];
		}
		temp = (tau * w[n + k] - alpha * vlag[k]) * sum;
		i2 = n;
		for (i = 1; i <= i2; ++i) {
			s[i] += temp * xpt[k + i * xpt_dim1];
		}
	}
	ss = 0;
	ds = 0;
	i2 = n;
	for (i = 1; i <= i2; ++i) {

		/* Computing 2nd power */
		d1 = s[i];
		ss += d1 * d1;
		ds += d[i] * s[i];
	}
	ssden = dd * ss - ds * ds;
	if (ssden >= dd * 1e-8 * ss) {
		goto BDL70;
	}

	/* Set the vector W before the RETURN from the subroutine. */
	BDL340: i2 = *ndim;
	for (k = 1; k <= i2; ++k) {
		w[k] = 0;
		for (j = 1; j <= 5; ++j) {
			w[k] += wvec[k + j * wvec_dim1] * par[j - 1];
		}
	}
	vlag[*kopt] += 1.;
}

void Newuoa::trsapp(long n, long npt, double *xopt, double *xpt, double *gq,
		double *hq, double *pq, double *delta, double *step, double *d,
		double *g, double *hd, double *hs, double *crvmin) {

	long xpt_dim1, xpt_offset, i1, i2, i, j, k, ih, iu, iterc, isave, itersw,
			itermax;
	double d1, d2, dd, cf, dg, gg, ds, sg, ss, dhd, dhs, cth, sgk, shs, sth,
			qadd, qbeg, qred, qmin, temp, qsav, qnew, ggbeg, alpha, angle,
			reduc, ggsav, delsq, tempa, tempb, bstep, ratio, twopi, angtest;

	/* Parameter adjustments */
	tempa = tempb = shs = sg = bstep = ggbeg = gg = qred = dd = 0.;
	xpt_dim1 = npt;
	xpt_offset = 1 + xpt_dim1;
	xpt -= xpt_offset;
	--xopt;
	--gq;
	--hq;
	--pq;
	--step;
	--d;
	--g;
	--hd;
	--hs;

	/* Functiontion Body */
	twopi = 2. * M_PI;
	delsq = *delta * *delta;
	iterc = 0;
	itermax = n;
	itersw = itermax;
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		d[i] = xopt[i];
	}
	goto TRL170;

	/* Prepare for the first line search. */
	TRL20: qred = dd = 0;
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		step[i] = 0;
		hs[i] = 0;
		g[i] = gq[i] + hd[i];
		d[i] = -g[i];

		/* Computing 2nd power */
		d1 = d[i];
		dd += d1 * d1;
	}
	*crvmin = 0;
	if (dd == 0) {
		goto TRL160;
	}
	ds = ss = 0;
	gg = dd;
	ggbeg = gg;

	/* Calculate the step to the trust region boundary and the product HD. */
	TRL40: ++iterc;
	temp = delsq - ss;
	bstep = temp / (ds + std::sqrt(ds * ds + dd * temp));
	goto TRL170;

	TRL50: dhd = 0;
	i1 = n;
	for (j = 1; j <= i1; ++j) {
		dhd += d[j] * hd[j];
	}

	/* Update CRVMIN and set the step-length ATRLPHA. */
	alpha = bstep;
	if (dhd > 0) {
		temp = dhd / dd;
		if (iterc == 1) {
			*crvmin = temp;
		}
		*crvmin = std::fmin(*crvmin, temp);

		/* Computing MIN */
		d1 = alpha, d2 = gg / dhd;
		alpha = std::fmin(d1, d2);
	}
	qadd = alpha * (gg - 0.5 * alpha * dhd);
	qred += qadd;

	/* Update STEP and HS. */
	ggsav = gg;
	gg = 0;
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		step[i] += alpha * d[i];
		hs[i] += alpha * hd[i];

		/* Computing 2nd power */
		d1 = g[i] + hs[i];
		gg += d1 * d1;
	}

	/* Begin another conjugate direction iteration if required. */
	if (alpha < bstep) {
		if (qadd <= qred * .01 || gg <= ggbeg * 1e-4 || iterc == itermax) {
			goto TRL160;
		}
		temp = gg / ggsav;
		dd = ds = ss = 0;
		i1 = n;
		for (i = 1; i <= i1; ++i) {
			d[i] = temp * d[i] - g[i] - hs[i];

			/* Computing 2nd power */
			d1 = d[i];
			dd += d1 * d1;
			ds += d[i] * step[i];

			/* Computing 2nd power */
			d1 = step[i];
			ss += d1 * d1;
		}
		if (ds <= 0) {
			goto TRL160;
		}
		if (ss < delsq) {
			goto TRL40;
		}
	}
	*crvmin = 0;
	itersw = iterc;

	/* Test whether an alternative iteration is required. */
	TRL90: if (gg <= ggbeg * 1e-4) {
		goto TRL160;
	}
	sg = 0;
	shs = 0;
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		sg += step[i] * g[i];
		shs += step[i] * hs[i];
	}
	sgk = sg + shs;
	angtest = sgk / std::sqrt(gg * delsq);
	if (angtest <= -.99) {
		goto TRL160;
	}

	/* Begin the alternative iteration by calculating D and HD and some
	 * scalar products. */
	++iterc;
	temp = std::sqrt(delsq * gg - sgk * sgk);
	tempa = delsq / temp;
	tempb = sgk / temp;
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		d[i] = tempa * (g[i] + hs[i]) - tempb * step[i];
	}
	goto TRL170;
	TRL120: dg = dhd = dhs = 0;
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		dg += d[i] * g[i];
		dhd += hd[i] * d[i];
		dhs += hd[i] * step[i];
	}

	/* Seek the value of the angle that minimizes Q. */
	cf = 0.5 * (shs - dhd);
	qbeg = sg + cf;
	qsav = qmin = qbeg;
	isave = 0;
	iu = 49;
	temp = twopi / (iu + 1.);
	i1 = iu;
	for (i = 1; i <= i1; ++i) {
		angle = 1. * i * temp;
		cth = std::cos(angle);
		sth = std::sin(angle);
		qnew = (sg + cf * cth) * cth + (dg + dhs * cth) * sth;
		if (qnew < qmin) {
			qmin = qnew;
			isave = i;
			tempa = qsav;
		} else if (i == isave + 1) {
			tempb = qnew;
		}
		qsav = qnew;
	}
	if (static_cast<double>(isave) == 0) {
		tempa = qnew;
	}
	if (isave == iu) {
		tempb = qbeg;
	}
	angle = 0;
	if (tempa != tempb) {
		tempa -= qmin;
		tempb -= qmin;
		angle = 0.5 * (tempa - tempb) / (tempa + tempb);
	}
	angle = temp * (1. * isave + angle);

	/* Calculate the new STEP and HS. Then test for convergence. */
	cth = std::cos(angle);
	sth = std::sin(angle);
	reduc = qbeg - (sg + cf * cth) * cth - (dg + dhs * cth) * sth;
	gg = 0;
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		step[i] = cth * step[i] + sth * d[i];
		hs[i] = cth * hs[i] + sth * hd[i];

		/* Computing 2nd power */
		d1 = g[i] + hs[i];
		gg += d1 * d1;
	}
	qred += reduc;
	ratio = reduc / qred;
	if (iterc < itermax && ratio > .01) {
		goto TRL90;
	}
	TRL160: return;

	/* The following instructions act as a subroutine for setting the
	 * vector HD to the vector D multiplied by the second derivative
	 * matrix of Q.  They are called from three different places, which
	 * are distinguished by the value of ITERC. */
	TRL170: i1 = n;
	for (i = 1; i <= i1; ++i) {
		hd[i] = 0;
	}
	i1 = npt;
	for (k = 1; k <= i1; ++k) {
		temp = 0;
		i2 = n;
		for (j = 1; j <= i2; ++j) {
			temp += xpt[k + j * xpt_dim1] * d[j];
		}
		temp *= pq[k];
		i2 = n;
		for (i = 1; i <= i2; ++i) {
			hd[i] += temp * xpt[k + i * xpt_dim1];
		}
	}
	ih = 0;
	i2 = n;
	for (j = 1; j <= i2; ++j) {
		i1 = j;
		for (i = 1; i <= i1; ++i) {
			++ih;
			if (i < j) {
				hd[j] += hq[ih] * d[i];
			}
			hd[i] += hq[ih] * d[j];
		}
	}
	if (iterc == 0) {
		goto TRL20;
	}
	if (iterc <= itersw) {
		goto TRL50;
	}
	goto TRL120;
}

void Newuoa::update(long n, long npt, double *bmat, double *zmat, long *idz,
		long *ndim, double *vlag, double *beta, long *knew, double *w) {

	long bmat_dim1, bmat_offset, zmat_dim1, zmat_offset, i1, i2, i, j, ja, jb,
			jl, jp, nptm, iflag;
	double d1, d2, tau, temp, scala, scalb, alpha, denom, tempa, tempb, tausq;

	/* Parameter adjustments */
	tempb = 0.;
	zmat_dim1 = npt;
	zmat_offset = 1 + zmat_dim1;
	zmat -= zmat_offset;
	bmat_dim1 = *ndim;
	bmat_offset = 1 + bmat_dim1;
	bmat -= bmat_offset;
	--vlag;
	--w;

	/* Functiontion Body */
	nptm = npt - n - 1;

	/* Apply the rotations that put zeros in the KNEW-th row of ZMAT. */
	jl = 1;
	i1 = nptm;
	for (j = 2; j <= i1; ++j) {
		if (j == *idz) {
			jl = *idz;
		} else if (zmat[*knew + j * zmat_dim1] != 0) {

			/* Computing 2nd power */
			d1 = zmat[*knew + jl * zmat_dim1];

			/* Computing 2nd power */
			d2 = zmat[*knew + j * zmat_dim1];
			temp = std::sqrt(d1 * d1 + d2 * d2);
			tempa = zmat[*knew + jl * zmat_dim1] / temp;
			tempb = zmat[*knew + j * zmat_dim1] / temp;
			i2 = npt;
			for (i = 1; i <= i2; ++i) {
				temp = tempa * zmat[i + jl * zmat_dim1]
						+ tempb * zmat[i + j * zmat_dim1];
				zmat[i + j * zmat_dim1] = tempa * zmat[i + j * zmat_dim1]
						- tempb * zmat[i + jl * zmat_dim1];
				zmat[i + jl * zmat_dim1] = temp;
			}
			zmat[*knew + j * zmat_dim1] = 0;
		}
	}

	/* Put the first NPT components of the KNEW-th column of HLAG into
	 * W, and calculate the parameters of the updating formula. */
	tempa = zmat[*knew + zmat_dim1];
	if (*idz >= 2) {
		tempa = -tempa;
	}
	if (jl > 1) {
		tempb = zmat[*knew + jl * zmat_dim1];
	}
	i1 = npt;
	for (i = 1; i <= i1; ++i) {
		w[i] = tempa * zmat[i + zmat_dim1];
		if (jl > 1) {
			w[i] += tempb * zmat[i + jl * zmat_dim1];
		}
	}
	alpha = w[*knew];
	tau = vlag[*knew];
	tausq = tau * tau;
	denom = alpha * *beta + tausq;
	vlag[*knew] -= 1.;

	/* Complete the updating of ZMAT when there is only 1.0 nonzero
	 * element in the KNEW-th row of the new matrix ZMAT, but, if IFLAG
	 * is set to 1.0, then the first column of ZMAT will be exchanged
	 * with another 1.0 later. */
	iflag = 0;
	if (jl == 1) {
		temp = std::sqrt((std::fabs(denom)));
		tempb = tempa / temp;
		tempa = tau / temp;
		i1 = npt;
		for (i = 1; i <= i1; ++i) {
			zmat[i + zmat_dim1] = tempa * zmat[i + zmat_dim1] - tempb * vlag[i];
		}
		if (*idz == 1 && temp < 0) {
			*idz = 2;
		}
		if (*idz >= 2 && temp >= 0) {
			iflag = 1;
		}
	} else {

		/* Complete the updating of ZMAT in the alternative case. */
		ja = 1;
		if (*beta >= 0) {
			ja = jl;
		}
		jb = jl + 1 - ja;
		temp = zmat[*knew + jb * zmat_dim1] / denom;
		tempa = temp * *beta;
		tempb = temp * tau;
		temp = zmat[*knew + ja * zmat_dim1];
		scala = 1. / std::sqrt(std::fabs(*beta) * temp * temp + tausq);
		scalb = scala * std::sqrt((std::fabs(denom)));
		i1 = npt;
		for (i = 1; i <= i1; ++i) {
			zmat[i + ja * zmat_dim1] = scala
					* (tau * zmat[i + ja * zmat_dim1] - temp * vlag[i]);
			zmat[i + jb * zmat_dim1] =
					scalb
							* (zmat[i + jb * zmat_dim1] - tempa * w[i]
									- tempb * vlag[i]);
		}
		if (denom <= 0) {
			if (*beta < 0) {
				++(*idz);
			}
			if (*beta >= 0) {
				iflag = 1;
			}
		}
	}

	/* IDZ is reduced in the following case, and usually the first
	 * column of ZMAT is exchanged with a later 1.0. */
	if (iflag == 1) {
		--(*idz);
		i1 = npt;
		for (i = 1; i <= i1; ++i) {
			temp = zmat[i + zmat_dim1];
			zmat[i + zmat_dim1] = zmat[i + *idz * zmat_dim1];
			zmat[i + *idz * zmat_dim1] = temp;
		}
	}

	/* Finally, update the matrix BMAT. */
	i1 = n;
	for (j = 1; j <= i1; ++j) {
		jp = npt + j;
		w[jp] = bmat[*knew + j * bmat_dim1];
		tempa = (alpha * vlag[jp] - tau * w[jp]) / denom;
		tempb = (-(*beta) * w[jp] - tau * vlag[jp]) / denom;
		i2 = jp;
		for (i = 1; i <= i2; ++i) {
			bmat[i + j * bmat_dim1] = bmat[i + j * bmat_dim1] + tempa * vlag[i]
					+ tempb * w[i];
			if (i > npt) {
				bmat[jp + (i - npt) * bmat_dim1] = bmat[i + j * bmat_dim1];
			}
		}
	}
	return;
}

double Newuoa::newuob(long n, long npt, double *x, double rhobeg, double rhoend,
		long maxfun, double *xbase, double *xopt, double *xnew, double *xpt,
		double *fval, double *gq, double *hq, double *pq, double *bmat,
		double *zmat, long *ndim, double *d, double *vlag, double *w,
		const multivariate &function, int &fev) {

	long xpt_dim1, xpt_offset, bmat_dim1, bmat_offset, zmat_dim1, zmat_offset,
			i1, i2, i3, i, j, k, ih, nf, nh, ip, jp, np, nfm, idz = 0, ipt, jpt,
			nfmm, knew = 0, kopt = 0, nptm, ksave, nfsav, itemp, ktemp, itest,
			nftest;
	double d1, d2, d3, f, dx, dsq, rho, sum, fbeg, diff, beta, gisq, temp, suma,
			sumb, fopt, bsum, gqsq, xipt, xjpt, sumz, diffa, diffb, diffc,
			hdiag, alpha, delta = 0., recip, reciq, fsave, dnorm, ratio, dstep,
			vquad, tempq, rhosq, detrat, crvmin = 0., distsq, xoptsq;

	/* Parameter adjustments */
	fev = 0;
	diffc = ratio = dnorm = nfsav = diffa = diffb = xoptsq = f = 0.;
	rho = fbeg = fopt = xjpt = xipt = 0.;
	itest = ipt = jpt = 0;
	alpha = dstep = 0.;
	zmat_dim1 = npt;
	zmat_offset = 1 + zmat_dim1;
	zmat -= zmat_offset;
	xpt_dim1 = npt;
	xpt_offset = 1 + xpt_dim1;
	xpt -= xpt_offset;
	--x;
	--xbase;
	--xopt;
	--xnew;
	--fval;
	--gq;
	--hq;
	--pq;
	bmat_dim1 = *ndim;
	bmat_offset = 1 + bmat_dim1;
	bmat -= bmat_offset;
	--d;
	--vlag;
	--w;

	/* Functiontion Body */
	np = n + 1;
	nh = n * np / 2;
	nptm = npt - np;
	nftest = (maxfun > 1) ? maxfun : 1;

	/* Set the initial elements of XPT, BMAT, HQ, PQ and ZMAT to 0. */
	i1 = n;
	for (j = 1; j <= i1; ++j) {
		xbase[j] = x[j];
		i2 = npt;
		for (k = 1; k <= i2; ++k) {
			xpt[k + j * xpt_dim1] = 0;
		}
		i2 = *ndim;
		for (i = 1; i <= i2; ++i) {
			bmat[i + j * bmat_dim1] = 0;
		}
	}
	i2 = nh;
	for (ih = 1; ih <= i2; ++ih) {
		hq[ih] = 0;
	}
	i2 = npt;
	for (k = 1; k <= i2; ++k) {
		pq[k] = 0;
		i1 = nptm;
		for (j = 1; j <= i1; ++j) {
			zmat[k + j * zmat_dim1] = 0;
		}
	}

	/* Begin the initialization procedure. NF becomes 1.0 more than the
	 * number of function values so far. The coordinates of the
	 * displacement of the next initial interpolation point from XBASE
	 * are set in XPT(NF,.). */
	rhosq = rhobeg * rhobeg;
	recip = 1. / rhosq;
	reciq = std::sqrt(.5) / rhosq;
	nf = 0;

	L50: nfm = nf;
	nfmm = nf - n;
	++nf;
	if (nfm <= n << 1) {
		if (nfm >= 1 && nfm <= n) {
			xpt[nf + nfm * xpt_dim1] = rhobeg;
		} else if (nfm > n) {
			xpt[nf + nfmm * xpt_dim1] = -(rhobeg);
		}
	} else {
		itemp = (nfmm - 1) / n;
		jpt = nfm - itemp * n - n;
		ipt = jpt + itemp;
		if (ipt > n) {
			itemp = jpt;
			jpt = ipt - n;
			ipt = itemp;
		}
		xipt = rhobeg;
		if (fval[ipt + np] < fval[ipt + 1]) {
			xipt = -xipt;
		}
		xjpt = rhobeg;
		if (fval[jpt + np] < fval[jpt + 1]) {
			xjpt = -xjpt;
		}
		xpt[nf + ipt * xpt_dim1] = xipt;
		xpt[nf + jpt * xpt_dim1] = xjpt;
	}

	/* Calculate the next value of F, label 70 being reached immediately
	 * after this calculation. The least function value so far and its
	 * index are required. */
	i1 = n;
	for (j = 1; j <= i1; ++j) {
		x[j] = xpt[nf + j * xpt_dim1] + xbase[j];
	}
	goto L310;

	L70: fval[nf] = f;
	if (nf == 1) {
		fbeg = fopt = f;
		kopt = 1;
	} else if (f < fopt) {
		fopt = f;
		kopt = nf;
	}

	/* Set the non0 initial elements of BMAT and the quadratic model
	 * in the cases when NF is at most 2*N+1. */
	if (nfm <= n << 1) {
		if (nfm >= 1 && nfm <= n) {
			gq[nfm] = (f - fbeg) / rhobeg;
			if (npt < nf + n) {
				bmat[nfm * bmat_dim1 + 1] = -1. / rhobeg;
				bmat[nf + nfm * bmat_dim1] = 1. / rhobeg;
				bmat[npt + nfm + nfm * bmat_dim1] = -.5 * rhosq;
			}
		} else if (nfm > n) {
			bmat[nf - n + nfmm * bmat_dim1] = .5 / rhobeg;
			bmat[nf + nfmm * bmat_dim1] = -.5 / rhobeg;
			zmat[nfmm * zmat_dim1 + 1] = -reciq - reciq;
			zmat[nf - n + nfmm * zmat_dim1] = reciq;
			zmat[nf + nfmm * zmat_dim1] = reciq;
			ih = nfmm * (nfmm + 1) / 2;
			temp = (fbeg - f) / rhobeg;
			hq[ih] = (gq[nfmm] - temp) / rhobeg;
			gq[nfmm] = .5 * (gq[nfmm] + temp);
		}

		/* Set the off-diagonal second derivatives of the Lagrange
		 * functions and the initial quadratic model. */
	} else {
		ih = ipt * (ipt - 1) / 2 + jpt;
		if (xipt < 0) {
			ipt += n;
		}
		if (xjpt < 0) {
			jpt += n;
		}
		zmat[nfmm * zmat_dim1 + 1] = recip;
		zmat[nf + nfmm * zmat_dim1] = recip;
		zmat[ipt + 1 + nfmm * zmat_dim1] = -recip;
		zmat[jpt + 1 + nfmm * zmat_dim1] = -recip;
		hq[ih] = (fbeg - fval[ipt + 1] - fval[jpt + 1] + f) / (xipt * xjpt);
	}
	if (nf < npt) {
		goto L50;
	}

	/* Begin the iterative procedure, because the initial model is
	 * complete. */
	rho = rhobeg;
	delta = rho;
	idz = 1;
	diffa = diffb = itest = xoptsq = 0;
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		xopt[i] = xpt[kopt + i * xpt_dim1];

		/* Computing 2nd power */
		d1 = xopt[i];
		xoptsq += d1 * d1;
	}

	L90: nfsav = nf;

	/* Generate the next trust region step and test its length. Set KNEW
	 * to -1 if the purpose of the next F will be to improve the
	 * model. */
	L100: knew = 0;
	trsapp(n, npt, &xopt[1], &xpt[xpt_offset], &gq[1], &hq[1], &pq[1], &delta,
			&d[1], &w[1], &w[np], &w[np + n], &w[np + (n << 1)], &crvmin);
	dsq = 0;
	i1 = n;
	for (i = 1; i <= i1; ++i) {

		/* Computing 2nd power */
		d1 = d[i];
		dsq += d1 * d1;
	}

	/* Computing MIN */
	d1 = delta, d2 = std::sqrt(dsq);
	dnorm = std::fmin(d1, d2);
	if (dnorm < .5 * rho) {
		knew = -1;
		delta = 0.1 * delta;
		ratio = -1.;
		if (delta <= rho * 1.5) {
			delta = rho;
		}
		if (nf <= nfsav + 2) {
			goto L460;
		}
		temp = crvmin * .125 * rho * rho;

		/* Computing MAX */
		d1 = std::fmax(diffa, diffb);
		if (temp <= std::fmax(d1, diffc)) {
			goto L460;
		}
		goto L490;
	}

	/* Shift XBASE if XOPT may be too far from XBASE. First make the
	 * changes to BMAT that do not depend on ZMAT. */
	L120: if (dsq <= xoptsq * .001) {
		tempq = xoptsq * .25;
		i1 = npt;
		for (k = 1; k <= i1; ++k) {
			sum = 0;
			i2 = n;
			for (i = 1; i <= i2; ++i) {
				sum += xpt[k + i * xpt_dim1] * xopt[i];
			}
			temp = pq[k] * sum;
			sum -= .5 * xoptsq;
			w[npt + k] = sum;
			i2 = n;
			for (i = 1; i <= i2; ++i) {
				gq[i] += temp * xpt[k + i * xpt_dim1];
				xpt[k + i * xpt_dim1] -= .5 * xopt[i];
				vlag[i] = bmat[k + i * bmat_dim1];
				w[i] = sum * xpt[k + i * xpt_dim1] + tempq * xopt[i];
				ip = npt + i;
				i3 = i;
				for (j = 1; j <= i3; ++j) {
					bmat[ip + j * bmat_dim1] = bmat[ip + j * bmat_dim1]
							+ vlag[i] * w[j] + w[i] * vlag[j];
				}
			}
		}

		/* Then the revisions of BMAT that depend on ZMAT are calculated. */
		i3 = nptm;
		for (k = 1; k <= i3; ++k) {
			sumz = 0;
			i2 = npt;
			for (i = 1; i <= i2; ++i) {
				sumz += zmat[i + k * zmat_dim1];
				w[i] = w[npt + i] * zmat[i + k * zmat_dim1];
			}
			i2 = n;
			for (j = 1; j <= i2; ++j) {
				sum = tempq * sumz * xopt[j];
				i1 = npt;
				for (i = 1; i <= i1; ++i) {
					sum += w[i] * xpt[i + j * xpt_dim1];
				}
				vlag[j] = sum;
				if (k < idz) {
					sum = -sum;
				}
				i1 = npt;
				for (i = 1; i <= i1; ++i) {
					bmat[i + j * bmat_dim1] += sum * zmat[i + k * zmat_dim1];
				}
			}
			i1 = n;
			for (i = 1; i <= i1; ++i) {
				ip = i + npt;
				temp = vlag[i];
				if (k < idz) {
					temp = -temp;
				}
				i2 = i;
				for (j = 1; j <= i2; ++j) {
					bmat[ip + j * bmat_dim1] += temp * vlag[j];
				}
			}
		}

		/* The following instructions complete the shift of XBASE,
		 * including the changes to the parameters of the quadratic model. */
		ih = 0;
		i2 = n;
		for (j = 1; j <= i2; ++j) {
			w[j] = 0;
			i1 = npt;
			for (k = 1; k <= i1; ++k) {
				w[j] += pq[k] * xpt[k + j * xpt_dim1];
				xpt[k + j * xpt_dim1] -= .5 * xopt[j];
			}
			i1 = j;
			for (i = 1; i <= i1; ++i) {
				++ih;
				if (i < j) {
					gq[j] += hq[ih] * xopt[i];
				}
				gq[i] += hq[ih] * xopt[j];
				hq[ih] = hq[ih] + w[i] * xopt[j] + xopt[i] * w[j];
				bmat[npt + i + j * bmat_dim1] = bmat[npt + j + i * bmat_dim1];
			}
		}
		i1 = n;
		for (j = 1; j <= i1; ++j) {
			xbase[j] += xopt[j];
			xopt[j] = 0;
		}
		xoptsq = 0;
	}

	/* Pick the model step if KNEW is positive. A different choice of D
	 * may be made later, if the choice of D by BIGLAG causes
	 * substantial cancellation in DENOM. */
	if (knew > 0) {
		biglag(n, npt, &xopt[1], &xpt[xpt_offset], &bmat[bmat_offset],
				&zmat[zmat_offset], &idz, ndim, &knew, &dstep, &d[1], &alpha,
				&vlag[1], &vlag[npt + 1], &w[1], &w[np], &w[np + n]);
	}

	/* Calculate VLAG and BETA for the current choice of D. The first
	 * NPT components of W_check will be held in W. */
	i1 = npt;
	for (k = 1; k <= i1; ++k) {
		suma = 0;
		sumb = 0;
		sum = 0;
		i2 = n;
		for (j = 1; j <= i2; ++j) {
			suma += xpt[k + j * xpt_dim1] * d[j];
			sumb += xpt[k + j * xpt_dim1] * xopt[j];
			sum += bmat[k + j * bmat_dim1] * d[j];
		}
		w[k] = suma * (.5 * suma + sumb);
		vlag[k] = sum;
	}
	beta = 0;
	i1 = nptm;
	for (k = 1; k <= i1; ++k) {
		sum = 0;
		i2 = npt;
		for (i = 1; i <= i2; ++i) {
			sum += zmat[i + k * zmat_dim1] * w[i];
		}
		if (k < idz) {
			beta += sum * sum;
			sum = -sum;
		} else {
			beta -= sum * sum;
		}
		i2 = npt;
		for (i = 1; i <= i2; ++i) {
			vlag[i] += sum * zmat[i + k * zmat_dim1];
		}
	}
	bsum = 0;
	dx = 0;
	i2 = n;
	for (j = 1; j <= i2; ++j) {
		sum = 0;
		i1 = npt;
		for (i = 1; i <= i1; ++i) {
			sum += w[i] * bmat[i + j * bmat_dim1];
		}
		bsum += sum * d[j];
		jp = npt + j;
		i1 = n;
		for (k = 1; k <= i1; ++k) {
			sum += bmat[jp + k * bmat_dim1] * d[k];
		}
		vlag[jp] = sum;
		bsum += sum * d[j];
		dx += d[j] * xopt[j];
	}
	beta = dx * dx + dsq * (xoptsq + dx + dx + .5 * dsq) + beta - bsum;
	vlag[kopt] += 1.;

	/* If KNEW is positive and if the cancellation in DENOM is
	 * unacceptable, then BIGDEN calculates an alternative model step,
	 * XNEW being used for working space. */
	if (knew > 0) {

		/* Computing 2nd power */
		d1 = vlag[knew];
		temp = 1. + alpha * beta / (d1 * d1);
		if (std::fabs(temp) <= .8) {
			bigden(n, npt, &xopt[1], &xpt[xpt_offset], &bmat[bmat_offset],
					&zmat[zmat_offset], &idz, ndim, &kopt, &knew, &d[1], &w[1],
					&vlag[1], &beta, &xnew[1], &w[*ndim + 1],
					&w[*ndim * 6 + 1]);
		}
	}

	/* Calculate the next value of the objective function. */
	L290: i2 = n;
	for (i = 1; i <= i2; ++i) {
		xnew[i] = xopt[i] + d[i];
		x[i] = xbase[i] + xnew[i];
	}
	++nf;

	L310: if (nf > nftest) {
		--nf;
		goto L530;
	}
	f = function(&x[1]);
	fev++;
	if (nf <= npt) {
		goto L70;
	}
	if (knew == -1) {
		goto L530;
	}

	/* Use the quadratic model to predict the change in F due to the
	 * step D, and set DIFF to the error of this prediction. */
	vquad = ih = 0;
	i2 = n;
	for (j = 1; j <= i2; ++j) {
		vquad += d[j] * gq[j];
		i1 = j;
		for (i = 1; i <= i1; ++i) {
			++ih;
			temp = d[i] * xnew[j] + d[j] * xopt[i];
			if (i == j) {
				temp = .5 * temp;
			}
			vquad += temp * hq[ih];
		}
	}
	i1 = npt;
	for (k = 1; k <= i1; ++k) {
		vquad += pq[k] * w[k];
	}
	diff = f - fopt - vquad;
	diffc = diffb;
	diffb = diffa;
	diffa = std::fabs(diff);
	if (dnorm > rho) {
		nfsav = nf;
	}

	/* Update FOPT and XOPT if the new F is the least value of the
	 * objective function so far. The branch when KNEW is positive
	 * occurs if D is not a trust region step. */
	fsave = fopt;
	if (f < fopt) {
		fopt = f;
		xoptsq = 0;
		i1 = n;
		for (i = 1; i <= i1; ++i) {
			xopt[i] = xnew[i];

			/* Computing 2nd power */
			d1 = xopt[i];
			xoptsq += d1 * d1;
		}
	}
	ksave = knew;
	if (knew > 0) {
		goto L410;
	}

	/* Pick the next value of DELTA after a trust region step. */
	if (vquad >= 0) {
		goto L530;
	}
	ratio = (f - fsave) / vquad;
	if (ratio <= 0.1) {
		delta = .5 * dnorm;
	} else if (ratio <= .7) {

		/* Computing MAX */
		d1 = .5 * delta;
		delta = std::fmax(d1, dnorm);
	} else {

		/* Computing MAX */
		d1 = .5 * delta, d2 = dnorm + dnorm;
		delta = std::fmax(d1, d2);
	}
	if (delta <= rho * 1.5) {
		delta = rho;
	}

	/* Set KNEW to the index of the next interpolation point to be deleted. */
	/* Computing MAX */
	d2 = 0.1 * delta;

	/* Computing 2nd power */
	d1 = std::fmax(d2, rho);
	rhosq = d1 * d1;
	ktemp = detrat = 0;
	if (f >= fsave) {
		ktemp = kopt;
		detrat = 1.;
	}
	i1 = npt;
	for (k = 1; k <= i1; ++k) {
		hdiag = 0;
		i2 = nptm;
		for (j = 1; j <= i2; ++j) {
			temp = 1.;
			if (j < idz) {
				temp = -1.;
			}

			/* Computing 2nd power */
			d1 = zmat[k + j * zmat_dim1];
			hdiag += temp * (d1 * d1);
		}

		/* Computing 2nd power */
		d2 = vlag[k];
		temp = (d1 = beta * hdiag + d2 * d2, std::fabs(d1));
		distsq = 0;
		i2 = n;
		for (j = 1; j <= i2; ++j) {

			/* Computing 2nd power */
			d1 = xpt[k + j * xpt_dim1] - xopt[j];
			distsq += d1 * d1;
		}
		if (distsq > rhosq) {

			/* Computing 3rd power */
			d1 = distsq / rhosq;
			temp *= d1 * (d1 * d1);
		}
		if (temp > detrat && k != ktemp) {
			detrat = temp;
			knew = k;
		}
	}
	if (knew == 0) {
		goto L460;
	}

	/* Update BMAT, ZMAT and IDZ, so that the KNEW-th interpolation
	 * point can be moved. Begin the updating of the quadratic model,
	 * starting with the explicit second derivative term. */
	L410: update(n, npt, &bmat[bmat_offset], &zmat[zmat_offset], &idz, ndim,
			&vlag[1], &beta, &knew, &w[1]);
	fval[knew] = f;
	ih = 0;
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		temp = pq[knew] * xpt[knew + i * xpt_dim1];
		i2 = i;
		for (j = 1; j <= i2; ++j) {
			++ih;
			hq[ih] += temp * xpt[knew + j * xpt_dim1];
		}
	}
	pq[knew] = 0;

	/* Update the other second derivative parameters, and then the
	 * gradient vector of the model. Also include the new interpolation
	 * point. */
	i2 = nptm;
	for (j = 1; j <= i2; ++j) {
		temp = diff * zmat[knew + j * zmat_dim1];
		if (j < idz) {
			temp = -temp;
		}
		i1 = npt;
		for (k = 1; k <= i1; ++k) {
			pq[k] += temp * zmat[k + j * zmat_dim1];
		}
	}
	gqsq = 0;
	i1 = n;
	for (i = 1; i <= i1; ++i) {
		gq[i] += diff * bmat[knew + i * bmat_dim1];

		/* Computing 2nd power */
		d1 = gq[i];
		gqsq += d1 * d1;
		xpt[knew + i * xpt_dim1] = xnew[i];
	}

	/* If a trust region step makes a small change to the objective
	 * function, then calculate the gradient of the least Frobenius norm
	 * interpolant at XBASE, and store it in W, using VLAG for a vector
	 * of right hand sides. */
	if (ksave == 0 && delta == rho) {
		if (std::fabs(ratio) > .01) {
			itest = 0;
		} else {
			i1 = npt;
			for (k = 1; k <= i1; ++k) {
				vlag[k] = fval[k] - fval[kopt];
			}
			gisq = 0;
			i1 = n;
			for (i = 1; i <= i1; ++i) {
				sum = 0;
				i2 = npt;
				for (k = 1; k <= i2; ++k) {
					sum += bmat[k + i * bmat_dim1] * vlag[k];
				}
				gisq += sum * sum;
				w[i] = sum;
			}

			/* Test whether to replace the new quadratic model by the
			 * least Frobenius norm interpolant, making the replacement
			 * if the test is satisfied. */
			++itest;
			if (gqsq < gisq * 100.) {
				itest = 0;
			}
			if (itest >= 3) {
				i1 = n;
				for (i = 1; i <= i1; ++i) {
					gq[i] = w[i];
				}
				i1 = nh;
				for (ih = 1; ih <= i1; ++ih) {
					hq[ih] = 0;
				}
				i1 = nptm;
				for (j = 1; j <= i1; ++j) {
					w[j] = 0;
					i2 = npt;
					for (k = 1; k <= i2; ++k) {
						w[j] += vlag[k] * zmat[k + j * zmat_dim1];
					}
					if (j < idz) {
						w[j] = -w[j];
					}
				}
				i1 = npt;
				for (k = 1; k <= i1; ++k) {
					pq[k] = 0;
					i2 = nptm;
					for (j = 1; j <= i2; ++j) {
						pq[k] += zmat[k + j * zmat_dim1] * w[j];
					}
				}
				itest = 0;
			}
		}
	}
	if (f < fsave) {
		kopt = knew;
	}

	/* If a trust region step has provided a sufficient decrease in F,
	 * then branch for another trust region calculation. The case
	 * KSAVE>0 occurs when the new function value was calculated by a
	 * model step. */
	if (f <= fsave + 0.1 * vquad) {
		goto L100;
	}
	if (ksave > 0) {
		goto L100;
	}

	/* Alternatively, find out if the interpolation points are close
	 * enough to the best point so far. */
	knew = 0;

	L460: distsq = delta * 4. * delta;
	i2 = npt;
	for (k = 1; k <= i2; ++k) {
		sum = 0;
		i1 = n;
		for (j = 1; j <= i1; ++j) {

			/* Computing 2nd power */
			d1 = xpt[k + j * xpt_dim1] - xopt[j];
			sum += d1 * d1;
		}
		if (sum > distsq) {
			knew = k;
			distsq = sum;
		}
	}

	/* If KNEW is positive, then set DSTEP, and branch back for the next
	 * iteration, which will generate a "model step". */
	if (knew > 0) {

		/* Computing MAX and MIN*/
		d2 = 0.1 * std::sqrt(distsq), d3 = .5 * delta;
		d1 = std::fmin(d2, d3);
		dstep = std::fmax(d1, rho);
		dsq = dstep * dstep;
		goto L120;
	}
	if (ratio > 0) {
		goto L100;
	}
	if (std::fmax(delta, dnorm) > rho) {
		goto L100;
	}

	/* The calculations with the current value of RHO are complete. Pick
	 * the next values of RHO and DELTA. */
	L490: if (rho > rhoend) {
		delta = .5 * rho;
		ratio = rho / rhoend;
		if (ratio <= 16.) {
			rho = rhoend;
		} else if (ratio <= 250.) {
			rho = std::sqrt(ratio) * rhoend;
		} else {
			rho = 0.1 * rho;
		}
		delta = std::fmax(delta, rho);
		goto L90;
	}

	/* Return from the calculation, after another Newton-Raphson step,
	 * if it is too short to have been tried before. */
	if (knew == -1) {
		goto L290;
	}

	L530: if (fopt <= f) {
		i2 = n;
		for (i = 1; i <= i2; ++i) {
			x[i] = xbase[i] + xopt[i];
		}
		f = fopt;
	}
	return f;
}

double Newuoa::newuoa(const multivariate &function, long n, long npt, double *x,
		double rhobeg, double rhoend, long maxfun, double *w, int &fev) {

	long id, np, iw, igq, ihq, ixb, ifv, ipq, ivl, ixn, ixo, ixp, ndim, nptm,
			ibmat, izmat;

	/* Parameter adjustments */
	--w;
	--x;

	/* Functiontion Body */
	np = n + 1;
	nptm = npt - np;
	ndim = npt + n;
	ixb = 1;
	ixo = ixb + n;
	ixn = ixo + n;
	ixp = ixn + n;
	ifv = ixp + n * npt;
	igq = ifv + npt;
	ihq = igq + n;
	ipq = ihq + n * np / 2;
	ibmat = ipq + npt;
	izmat = ibmat + ndim * n;
	id = izmat + npt * nptm;
	ivl = id + n;
	iw = ivl + ndim;

	/* The above settings provide a partition of W for subroutine
	 * NEWUOB. The partition requires the first NPT*(NPT+N)+5*N*(N+3)/2
	 * elements of W plus the space that is needed by the last array of
	 * NEWUOB. */
	return newuob(n, npt, &x[1], rhobeg, rhoend, maxfun, &w[ixb], &w[ixo],
			&w[ixn], &w[ixp], &w[ifv], &w[igq], &w[ihq], &w[ipq], &w[ibmat],
			&w[izmat], &ndim, &w[id], &w[ivl], &w[iw], function, fev);
}
