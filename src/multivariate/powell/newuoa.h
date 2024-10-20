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

#ifndef MULTIVARIATE_NEWUOA_H_
#define MULTIVARIATE_NEWUOA_H_

#include "../multivariate.h"

class Newuoa: public MultivariateOptimizer {

protected:
	int _np, _mfev;
	double _rho, _tol;

public:
	Newuoa(int mfev, int np, double rho, double tol);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void biglag(long n, long npt, double *xopt, double *xpt, double *bmat,
			double *zmat, long *idz, long *ndim, long *knew, double *delta,
			double *d, double *alpha, double *hcol, double *gc, double *gd,
			double *s, double *w);

	void bigden(long n, long npt, double *xopt, double *xpt, double *bmat,
			double *zmat, long *idz, long *ndim, long *kopt, long *knew,
			double *d, double *w, double *vlag, double *beta, double *s,
			double *wvec, double *prod);

	void trsapp(long n, long npt, double *xopt, double *xpt, double *gq,
			double *hq, double *pq, double *delta, double *step, double *d,
			double *g, double *hd, double *hs, double *crvmin);

	void update(long n, long npt, double *bmat, double *zmat, long *idz,
			long *ndim, double *vlag, double *beta, long *knew, double *w);

	double newuob(long n, long npt, double *x, double rhobeg, double rhoend,
			long maxfun, double *xbase, double *xopt, double *xnew, double *xpt,
			double *fval, double *gq, double *hq, double *pq, double *bmat,
			double *zmat, long *ndim, double *d, double *vlag, double *w,
			const multivariate &function, int &fev);

	double newuoa(const multivariate &function, long n, long npt, double *x,
			double rhobeg, double rhoend, long maxfun, double *w, int &fev);
};

#endif /* MULTIVARIATE_NEWUOA_H_ */
