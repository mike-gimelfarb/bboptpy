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

#ifndef MULTIVARIATE_BOBYQA_H_
#define MULTIVARIATE_BOBYQA_H_

#include "../multivariate.h"

class Bobyqa: public MultivariateOptimizer {

protected:
	int _np, _mfev;
	double _rho, _tol;

public:
	Bobyqa(int mfev, int np, double rho, double tol);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void altmov(const long n, const long npt, const double *xpt,
			const double *const xopt, const double *bmat, const double *zmat,
			const long ndim, const double *const sl, const double *const su,
			const long kopt, const long knew, const double adelt,
			double *const xnew, double *const xalt, double &alpha,
			double &cauchy, double *const glag, double *const hcol,
			double *const w);

	void prelim(const multivariate &function, const long n, const long npt,
			double *const x, const double *const xl, const double *const xu,
			const double rhobeg, const long maxfun, double *const xbase,
			double *xpt, double *const fval, double *const gopt,
			double *const hq, double *const pq, double *bmat, double *zmat,
			const long ndim, const double *const sl, const double *const su,
			long &nf, long &kopt, long &fev);

	void rescue(const multivariate &function, const long n, const long npt,
			const double *const xl, const double *const xu, const long maxfun,
			double *const xbase, double *xpt, double *const fval,
			double *const xopt, double *const gopt, double *const hq,
			double *const pq, double *bmat, double *zmat, const long ndim,
			double *const sl, double *const su, long &nf, const double delta,
			long &kopt, double *const vlag, double *const ptsaux,
			double *const ptsid, double *const w, long &fev);

	void trsbox(const long n, const long npt, const double *xpt,
			const double *const xopt, const double *const gopt,
			const double *const hq, const double *const pq,
			const double *const sl, const double *const su, const double delta,
			double *const xnew, double *const d, double *const gnew,
			double *const xbdi, double *const s, double *const hs,
			double *const hred, double *const dsq, double *const crvmin);

	void update(const long n, const long npt, double *bmat, double *zmat,
			const long ndim, double *const vlag, const double beta,
			const double denom, const long knew, double *const w);

	double bobyqb(const multivariate &function, const long n, const long npt,
			double *const x, const double *const xl, const double *const xu,
			const double rhobeg, const double rhoend, const long maxfun,
			double *const xbase, double *xpt, double *const fval,
			double *const xopt, double *const gopt, double *const hq,
			double *const pq, double *bmat, double *zmat, const long ndim,
			double *const sl, double *const su, double *const xnew,
			double *const xalt, double *const d, double *const vlag,
			double *const w, long &fev);

	double bobyqa(const multivariate &function, const long n, const long npt,
			double *x, const double *xl, const double *xu, const double rhobeg,
			const double rhoend, const long maxfun, double *w, long &fev);
};

#endif /* MULTIVARIATE_BOBYQA_H_ */
