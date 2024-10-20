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

 [1] Glasmachers, Tobias, and Oswin Krause. "The Hessian Estimation Evolution Strategy."
 International Conference on Parallel Problem Solving from Nature. Springer, Cham, 2020.
 */

#ifndef MULTIVARIATE_HEES_H_
#define MULTIVARIATE_HEES_H_

#include <memory>
#include <random>

#include "../../tabular.hpp"

#include "../multivariate.h"

struct hees_index {

	int _index, _rank;
	double _value;

	static bool compare_fitness(const hees_index &x, const hees_index &y) {
		return x._value < y._value;
	}

	static bool compare_index(const hees_index &x, const hees_index &y) {
		return x._index < y._index;
	}
};

class Hees: public MultivariateOptimizer {

protected:
	bool _adaptpop, _print;
	int _fev, _mfev, _n, _B, _mu, _np, _mres;
	double _tol, _fm, _mueff, _mueffm, _cs, _ds, _chi, _gs, _kappa, _etaA,
			_sigma0, _sigma, _fbest;
	multivariate_problem _f;
	Tabular _table;
	std::vector<double> _lower, _upper, _xmean, _weights, _norms, _hess, _q,
			_dz, _ps, _xbest;
	std::vector<std::vector<double>> _a, _aold, _b, _x, _g;
	std::vector<hees_index> _fitness;

public:
	std::normal_distribution<> _Z { 0., 1. };

	Hees(int mfev, double tol, int mres = 1, bool print = false, int np = 0,
			double sigma0 = 2.);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void samplePopulation();

	void evaluateAndSortPopulation();

	void covarianceUpdate();

	void meanUpdate();

	void stepSizeUpdate();

	bool converged();
};

#endif /* MULTIVARIATE_HEES_H_ */
