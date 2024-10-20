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

 [1] Birgin, Ernesto G., and José Mario Martínez. "Improving ultimate convergence
 of an augmented Lagrangian method." Optimization Methods and Software 23.2 (2008):
 177-195.
 */

#ifndef MULTIVARIATE_ALGENCAN_H_
#define MULTIVARIATE_ALGENCAN_H_

#include <vector>

#include "../../string_utils.h"
#include "../../tabular.hpp"
#include "../multivariate.h"

class Algencan: public MultivariateOptimizer {

protected:
	bool _print, _localconv;
	int _n, _m, _p, _it, _fev, _cev, _bbev, _mit;
	double _tol, _rho, _rhoold, _gamma, _tau, _holdnorm, _voldnorm, _lambdamin,
			_lambdamax, _mumax, _lambda0, _mu0, _icmbest;
	multivariate_problem _f;
	MultivariateOptimizer *_local;
	Tabular _table;
	std::vector<double> _lower, _upper, _x0, _ghx, _mu, _lambda, _xbest;

public:
	Algencan(MultivariateOptimizer *local, int mit, double tol, bool print =
			false, double tau = 0.5, double gamma = 10., double lambda0 = 0.,
			double mu0 = 0.);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

	double lagrangian(const double *x);

private:
	double initialRho(const double *x);

	void solveLocal();

	void updateRho();

	void updateMultipliers();

	double icm();
};

#endif /* MULTIVARIATE_ALGENCAN_H_ */
