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
 */

#ifndef MULTIVARIATE_BASE_CMAES_H_
#define MULTIVARIATE_BASE_CMAES_H_

#include <memory>

#include "../multivariate.h"

struct cmaes_index {

	int _index;
	double _value;

	static bool compare_fitness(const cmaes_index &x, const cmaes_index &y) {
		return x._value < y._value;
	}
};

struct cmaes_history {

	int _cap, _buffer, _len;
	std::vector<double> _hist;

	void add(double value);

	double get(int i);
};

class BaseCmaes: public MultivariateOptimizer {

protected:
	bool _adaptpop, _adaptit, _bound;
	int _n, _mfev, _mit, _lambda, _mu, _it, _hlen, _ik, _fev;
	double _tol, _sigma0, _mueff, _cc, _cs, _c1, _cmu, _damps, _chi, _sigma,
			_fbest, _fworst;
	multivariate_problem _f;
	cmaes_history _best, _kth;
	std::vector<int> _ibw;
	std::vector<double> _lower, _upper, _ybw, _xmean, _xold, _weights, _artmp,
			_pc, _ps;
	std::vector<std::vector<double>> _arx;
	std::vector<cmaes_index> _fitness;

public:
	BaseCmaes(int mfev, double tol, int np, double sigma0 = 2., bool bound =
			false);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

	void setParams(int np, double sigma, int mfev);

	virtual void samplePopulation() = 0;

	virtual void updateDistribution() = 0;

	virtual bool converged() = 0;

	virtual void updateSigma();

	virtual void updateHistory();

	virtual void evaluateAndSortPopulation();

	std::vector<double> bestSolution();
};

#endif /* MULTIVARIATE_BASE_CMAES_H_ */
