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

#ifndef MULTIVARIATE_ROSENBROCK_H_
#define MULTIVARIATE_ROSENBROCK_H_

#include "../multivariate.h"

class Rosenbrock: public MultivariateOptimizer {

protected:
	int _n, _mfev, _fev, _i, _ierr;
	double _tol, _rho, _step0, _stepi, _wa;
	multivariate_problem _f;
	std::vector<double> _lower, _upper, _x0, _x1, _d, _temp, _fs, _wx;
	std::vector<std::vector<double>> _v, _vold, _x;

public:
	Rosenbrock(int mfev, double tol, double step0, double decf = 0.1);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	int lineSearch(const multivariate &f, int n, double *pos, double &s,
			double *v, double *x, double *x0, double *fs, int &fev, int mfev);
};

#endif /* MULTIVARIATE_ROSENBROCK_H_ */
