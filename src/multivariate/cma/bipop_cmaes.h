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

 [1] Hansen, Nikolaus. "Benchmarking a BI-population CMA-ES on the BBOB-2009
 function testbed." Proceedings of the 11th Annual Conference Companion on
 Genetic and Evolutionary Computation Conference: Late Breaking Papers. ACM,
 2009.

 [2] Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box
 Optimization Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the
 BBOB-2012 Noiseless Testbed." Genetic and Evolutionary Computation Conference
 (GECCO-2012), ACM Press : 269-276. July 2012.

 [3] Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Alternative
 restart strategies for CMA-ES." International Conference on Parallel Problem
 Solving from Nature. Springer, Berlin, Heidelberg, 2012.
 */

#ifndef MULTIVARIATE_BIPOP_CMAES_H_
#define MULTIVARIATE_BIPOP_CMAES_H_

#include <random>

#include "../../tabular.hpp"

#include "base_cmaes.h"

class BiPopCmaes: public MultivariateOptimizer {

protected:
	bool _print, _nbipop;
	int _n, _mfev, _fev, _maxlargeruns, _lambdadef, _largelambda, _smalllambda,
			_largebudget, _smallbudget, _largerestarts, _smallrestarts, _it,
			_bestregime;
	double _fx, _sigmadef, _largesigma, _smallsigma, _fxold, _fxbest,
			_ksigmadec, _kbudget;
	multivariate_problem _f;
	BaseCmaes *_base;
	Tabular _table;
	std::vector<double> _lower, _upper, _guess, _x0, _xbest;

public:
	std::normal_distribution<> _Z { 0., 1. };

	BiPopCmaes(BaseCmaes *base, int mfev, bool print = false,
			double sigma0 = 2., int maxlargeruns = 9, bool nbipop = true,
			double ksigmadec = 1.6, double kbudget = 2.);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	int computeMaxEvaluations(const int lambda);

	void runFirstRegime();

	void runSecondRegime();
};

#endif /* MULTIVARIATE_BIPOP_CMAES_H_ */
