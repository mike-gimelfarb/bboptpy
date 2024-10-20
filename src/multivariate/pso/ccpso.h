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

 [1] Van den Bergh, Frans, and Andries Petrus Engelbrecht. "A cooperative approach
 to particle swarm optimization." IEEE transactions on evolutionary computation
 8.3 (2004): 225-239.

 [2] Li, Xiaodong, and Xin Yao. "Cooperatively coevolving particle swarms for
 large scale optimization." IEEE Transactions on Evolutionary Computation 16.2
 (2012): 210-224.

 [3] Li, Xiaodong, and Xin Yao. "Tackling high dimensional nonseparable optimization
 problems by cooperatively coevolving particle swarms." 2009 IEEE congress on
 evolutionary computation. IEEE, 2009.
 */

#ifndef MULTIVARIATE_PSO_CCPSO_H_
#define MULTIVARIATE_PSO_CCPSO_H_

#include <memory>
#include <random>
#include <vector>

#include "../multivariate.h"

class CCPSOSearch: public MultivariateOptimizer {

protected:
	bool _correct, _improved, _adaptp;
	int _n, _fev, _gen, _np, _npps, _mfev, _nswarm, _cpswarm, _is, _localfreq;
	double _stol, _fyhat, _phat0, _phat;
	multivariate_problem _f;
	MultivariateOptimizer *_local;
	std::vector<int> _pps, _idx3, _range;
	std::vector<double> _lower, _upper, _temp, _yhat, _z, _temp3, _wlb, _wub,
			_wguess;
	std::vector<std::vector<int>> _k, _ibest, _strat;
	std::vector<std::vector<double>> _X, _Y, _fX, _fY;

public:
	std::normal_distribution<> _Z { 0., 1. };

	CCPSOSearch(int mfev, double sigmatol, int np, int *pps, int npps,
			bool correct = true, double pcauchy = -1.,
			MultivariateOptimizer *local = nullptr, int localfreq = 10);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();

	double evaluate(int i, double *z);

	void randomizeComponents();

	void updateSwarm();

	void updatePosition();

	void localSearch();

	double sampleCauchy();

	int sampleSubsetIndex();
};

#endif /* MULTIVARIATE_PSO_CCPSO_H_ */
