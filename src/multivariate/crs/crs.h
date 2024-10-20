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

 [1] Kaelo, P., and M. M. Ali. "Some variants of the controlled random search
 algorithm for global optimization." Journal of optimization theory and
 applications 130.2 (2006): 253-264.
 */

#ifndef MULTIVARIATE_CRS_CRS_H_
#define MULTIVARIATE_CRS_CRS_H_

#include <random>

#include "../multivariate.h"

class CrsSearch: public MultivariateOptimizer {

protected:
	int _n, _np, _fev, _mfev;
	double _tol, _ftrial, _ftrial2;
	multivariate_problem _f;
	std::vector<double> _lower, _upper, _centroid, _trial, _trial2;
	std::vector<point> _points;

public:
	CrsSearch(int mfev, int np, double tol);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

protected:
	void trial();

	void update();

	bool stop();

	bool inBounds(double *p);
};

#endif /* MULTIVARIATE_CRS_CRS_H_ */
