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

 [1] Auger, Anne, and Nikolaus Hansen. "A restart CMA evolution strategy with
 increasing population size." Evolutionary Computation, 2005. The 2005 IEEE
 Congress on. Vol. 2. IEEE, 2005.

 [2] Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box
 Optimization Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the
 BBOB-2012 Noiseless Testbed." Genetic and Evolutionary Computation Conference
 (GECCO-2012), ACM Press : 269-276. July 2012.

 [3] Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Alternative
 restart strategies for CMA-ES." International Conference on Parallel Problem
 Solving from Nature. Springer, Berlin, Heidelberg, 2012.

 [4] Liao, Tianjun, and Thomas Stützle. "Bounding the population size of
 IPOP-CMA-ES on the noiseless BBOB testbed." Proceedings of the 15th annual
 conference companion on Genetic and evolutionary computation. ACM, 2013.

 [5] https://coco.gforge.inria.fr/lib/exe/fetch.php?media=slides2013:00
 -bbob2013slides.pdf
 */

#ifndef MULTIVARIATE_IPOP_CMAES_H_
#define MULTIVARIATE_IPOP_CMAES_H_

#include <random>

#include "../../tabular.hpp"

#include "base_cmaes.h"

class IPopCmaes: public MultivariateOptimizer {

protected:
	bool _print, _nipop, _boundlambda;
	int _n, _mfev, _it, _fev, _lambdadef, _lambdamax, _lambda;
	double _sigmadef, _sigma, _ksigmadec, _fbest, _fx;
	multivariate_problem _f;
	BaseCmaes *_base;
	Tabular _table;
	std::vector<double> _lower, _upper, _guess, _xbest, _x, _x0;

public:
	std::normal_distribution<> _Z { 0., 1. };

	IPopCmaes(BaseCmaes *base, int mfev, bool print = false, double sigma0 = 2.,
			bool nipop = true, double ksigmadec = 1.6, bool boundlambda = true);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	int computeMaxEvaluations(const int lambda);

};

#endif /* MULTIVARIATE_IPOP_CMAES_H_ */
