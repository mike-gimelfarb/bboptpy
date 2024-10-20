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

 [1] Tanabe, Ryoji, and Alex Fukunaga. "Success-history based parameter
 adaptation for differential evolution." 2013 IEEE congress on evolutionary
 computation. IEEE, 2013.

 [2] Tanabe, Ryoji, and Alex S. Fukunaga. "Improving the search performance of
 SHADE using linear population size reduction." 2014 IEEE congress on
 evolutionary computation (CEC). IEEE, 2014.
 */

#ifndef MULTIVARIATE_DE_SHADE_H_
#define MULTIVARIATE_DE_SHADE_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class ShadeSearch: public MultivariateOptimizer {

protected:
	bool _archive, _repaircr;
	int _n, _npinit, _npmin, _np, _fev, _mfev, _h, _k;
	double _tol;
	multivariate_problem _f;
	std::vector<double> _lower, _upper, _MCR, _MF, _work, _SCR, _SF, _w;
	std::vector<std::vector<double>> _arch;
	std::vector<point> _swarm;

public:
	std::normal_distribution<> _Z { 0., 1. };

	ShadeSearch(int mfev, int npinit, double tol, bool archive = true,
			bool repaircr = true, int h = 100, int npmin = 4);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();

	double mutate(double *x, double *best, double *xr1, double *xr2,
			double *out, int n, double F, double CR);

	double sampleCauchy();
};

#endif /* MULTIVARIATE_DE_SHADE_H_ */
