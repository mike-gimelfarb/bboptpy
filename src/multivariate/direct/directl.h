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

#ifndef MULTIVARIATE_DIRECTL_H_
#define MULTIVARIATE_DIRECTL_H_

#include "../multivariate.h"

class Directl: public MultivariateOptimizer {

protected:
	int _n, _mfev, _method, _jones, _lmaxdim, _maxdeep, _maxdiv;
	double _eps, _volper, _sigmaper, _fglobal, _fglper;
	multivariate_problem _f;
	std::vector<double> _lower, _upper;

	int _free, _t, _actdeep, _minpos, _maxpos, _help, _numfunc, _maxi, _oops,
			_cheat, _actmaxdeep, _oldpos, _tstart, _start, _newtosample, _pos1,
			_ifeas, _iifeas, _mdeep, _oldmaxf, _inc, _ierror, _actdeepdiv,
			_iepschange, _maxf;
	double _divfactor, _kmax, _delta, _epsfix, _fmin, _fmax;
	std::vector<int> _anchor, _S, _point, _length, _arrayi, _list2;
	std::vector<double> _c, _thirds, _levels, _w, _l, _u, _farr, _x;

public:
	Directl(int mfev, double volper, double sigmaper, double eps = 0.,
			int method = 0);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void dirInit(double *f, const multivariate &fcn, double *c, int *length,
			int &actdeep, int *point, int *anchor, int &free, int *arrayi,
			int &maxi, int *list2, double *w, double *x, double *l, double *u,
			double &fmin, int &minpos, double *thirds, double *levels,
			int maxfunc, int maxdeep, int n, double &fmax, int &ifeas,
			int &iifeas, int &ierr);

	void dirHeader(int n, double &eps, int maxf, double *lower, double *upper,
			int maxfunc, int &ierror, double &epsfix, int &iepschange);

	void dirSampleF(double *c, int *arrayi, int nnew, int *length, double *f,
			int maxi, int *point, const multivariate &fcn, double *x, double *l,
			double &fmin, int &minpos, double *u, int n, int maxfunc,
			double &fmax, int &ifeas, int &iifeas);

	void dirChoose(int *anchor, int *S, int actdeep, double *f, double fmin,
			double eps, double *thirds, int &maxpos, int *length, int maxfunc,
			int maxdeep, int maxdiv, int n, int cheat, double kmax, int ifeas);

	void dirDoubleInsert(int *anchor, int *S, int maxpos, int *point, double *f,
			int maxfunc, int maxdiv, int &ierror);

	void dirReplaceInf(int free, double *f, double *c, double *thirds,
			int *length, int *anchor, int *point, double *c1, double *c2,
			int maxfunc, int maxdeep, int n, double fmax);

	int dirGetMaxDeep(int pos, int *length, int maxfunc, int n);

	void dirResortList(int replace, int *anchor, double *f, int *point,
			int *length, int n, int maxfunc);

	void dirInsertList(int &nnew, int *anchor, int *point, double *f, int maxi,
			int *length, int maxfunc, int maxdeep, int n, int samp);

	int dirGetLevel(int pos, int *length, int maxfunc, int n);

	void dirDivide(int nnew, int clen, int *length, int *point, int *arrayi,
			int sample, int *list2, double *w, int maxi, double *f, int maxfunc,
			int maxdeep, int n);

	void dirInsertList2(int &start, int j, int k, int *list2, double *w,
			int maxi, int n);

	void dirSearchMin(int &start, int *list2, int &pos, int &k, int n);

	void dirSamplePoints(double *c, int *arrayi, double delta, int sample,
			int &start, int *length, int &free, int maxi, int *point, int n,
			int maxfunc, int &oops);

	void dirGetI(int *length, int pos, int *arrayi, int &maxi, int n,
			int maxfunc);

	void dirInitList(int *anchor, int &free, int *point, double *f, int maxfunc,
			int maxdeep);

	void dirInsert3(int &pos1, int &pos2, int &pos3, int deep, int *anchor,
			int *point, int free, double *f, double &fmin, int &minpos,
			int maxfunc);

	void dirInsert(int &start, int ins, int *point, double *f, int maxfunc);

	void dirSort3(int &pos1, int &pos2, int &pos3, double *f);

	int dirPreprc(double *u, double *l, int n, double *xs1, double *xs2);

	double dirInfcn(const multivariate &f, double *x, double *c1, double *c2,
			int n);

	int isInBox(double *x, double *a, double *b, int n, int lmd);
};

#endif /* MULTIVARIATE_DIRECTL_H_ */
