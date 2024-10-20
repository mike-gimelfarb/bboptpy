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

 [1] Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Adaptive coordinate
 descent." Proceedings of the 13th annual conference on Genetic and evolutionary
 computation. 2011.

 [2] Hansen, Nikolaus. "Adaptive encoding: How to render search coordinate system
 invariant." International Conference on Parallel Problem Solving from Nature.
 Springer, Berlin, Heidelberg, 2008.
 */

#ifndef MULTIVARIATE_ACD_H_
#define MULTIVARIATE_ACD_H_

#include "../multivariate.h"

class ACD: public MultivariateOptimizer {

protected:
	bool _improved;
	int _n, _fev, _mfev, _ix, _it, _itae, _updateperiod, _convergeperiod;
	double _ftol, _xtol, _ksucc, _kunsucc, _c1, _cmu, _cp, _fbest;
	multivariate_problem _f;
	std::vector<int> _order;
	std::vector<double> _lower, _upper, _xbest, _sigma, _x1, _x2, _weights, _p,
			_m, _mold, _diagd, _artmp, _fhist;
	std::vector<std::vector<double>> _b, _invB, _c;
	std::vector<point> _points;

public:
	ACD(int mfev, double ftol, double xtol, double ksucc = 2., double kunsucc =
			0.5);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();

	void updateAE();

	void tred2();

	void tql2();
};

#endif /* MULTIVARIATE_ACD_H_ */
