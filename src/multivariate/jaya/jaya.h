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

 [1] Rao, R. "Jaya: A simple and new optimization algorithm for solving
 constrained and unconstrained optimization problems." International Journal
 of Industrial Engineering Computations 7.1 (2016): 19-34.

 [2] Rao, R. Venkata, and Ankit Saroj. "A self-adaptive multi-population based
 Jaya algorithm for engineering optimization." Swarm and Evolutionary computation
 37 (2017): 1-26.

 [3] Yu, Jiang-Tao, et al. "Jaya algorithm with self-adaptive multi-population
 and Lévy flights for solving economic load dispatch problems." IEEE Access 7
 (2019): 21372-21384.

 [4] Ravipudi, Jaya Lakshmi, and Mary Neebha. "Synthesis of linear antenna arrays
 using jaya, self-adaptive jaya and chaotic jaya algorithms." AEU-International
 Journal of Electronics and Communications 92 (2018): 54-63.
 */

#ifndef MULTIVARIATE_JAYA_H_
#define MULTIVARIATE_JAYA_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class JayaSearch: public MultivariateOptimizer {

public:
	enum jaya_mutation_method {
		original, levy, tent_map, logistic
	};

protected:
	bool _adapt_k;
	int _n, _fev, _k, _np, _npmin, _mfev, _k0, _nks, _kcheb;
	double _pbest, _best, _fgbest, _sigmau, _scale, _beta, _temper, _stol,
			_xchaos;
	jaya_mutation_method _mutation;
	multivariate_problem _f;
	std::vector<int> _len;
	std::vector<double> _lower, _upper, _tmp, _bestx, _perfindex, _pstrat;
	std::vector<point> _pool;

public:
	std::normal_distribution<> _Z { 0., 1. };

	JayaSearch(int mfev, double tol, int np, int npmin, bool adapt = true,
			int k0 = 2, jaya_mutation_method mutation = logistic, double scale =
					0.01, double beta = 1.5, int kcheb = 2,
			double temper = 10.);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();

	void divideSubpopulation();

	void adaptK();

	void evolve(int i, point &best, point &worst);

	double sampleLevy();

	double sampleTentMap();

	double sampleLogistic();
};

#endif /* MULTIVARIATE_JAYA_H_ */
