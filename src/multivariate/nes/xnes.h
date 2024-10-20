/*
 Copyright (c) 2024 Mike Gimelfarb

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

 [1] Glasmachers, T., Schaul, T., Yi, S., Wierstra, D., & Schmidhuber, J.
 (2010, July). Exponential natural evolution strategies. In Proceedings of the
 12th annual conference on Genetic and evolutionary computation (pp. 393-400).
 */

#ifndef MULTIVARIATE_XNES_H_
#define MULTIVARIATE_XNES_H_

#include <memory>
#include <random>

#include "../multivariate.h"

struct xnes_point {

	double _f;
	std::vector<double> _x, _z;

	static bool compare_fitness(const xnes_point &x, const xnes_point &y) {
		return x._f < y._f;
	}
};

class xNES: public MultivariateOptimizer {

protected:
	int _n, _mfev, _np, _fev;
	double _tol, _a0, _etamu, _etasigma, _etab, _sigma, _Gsigma;
	multivariate_problem _f;
	std::vector<double> _lower, _upper, _u, _mu, _Gdelta, _diagd, _artmp;
	std::vector<std::vector<double>> _B, _G, _b, _c;
	std::vector<xnes_point> _points;

public:
	std::normal_distribution<> _Z { 0., 1. };

	xNES(int mfev, double tol, double a0 = 1., double etamu = 1.);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();

	void exponential();

	void tred2();

	void tql2();
};

#endif /* MULTIVARIATE_XNES_H_ */
