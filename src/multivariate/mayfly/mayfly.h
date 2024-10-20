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

 [1] Zervoudakis, Konstantinos, and Stelios Tsafarakis. "A mayfly optimization algorithm."
 Computers & Industrial Engineering 145 (2020): 106559.
 */

#ifndef MULTIVARIATE_MAYFLY_H_
#define MULTIVARIATE_MAYFLY_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class MayflySearch: public MultivariateOptimizer {

	struct mayfly {

		std::vector<double> _x, _v, _bx;
		double _f, _bf;

		static bool compare_fitness(const std::shared_ptr<mayfly> &x,
				const std::shared_ptr<mayfly> &y) {
			return x->_f < y->_f;
		}
	};

protected:
	bool _pgb;
	int _n, _np, _fev, _mfev, _nmut, _it, _itmax;
	double _a1, _a2, _a3, _beta, _dance0, _ddamp, _dance, _fl0, _fldamp, _fl,
			_gmin, _gmax, _g, _fbest, _vdamp, _sigma, _mu, _pmut, _l;
	multivariate_problem _f;
	std::vector<int> _indices;
	std::vector<double> _lower, _upper, _best;
	std::vector<std::shared_ptr<mayfly>> _males, _females, _offspring, _mutated,
			_temp;

public:
	std::normal_distribution<> _Z { 0., 1. };

	MayflySearch(int np, int mfev, double a1 = 1., double a2 = 1.5, double a3 =
			1.5, double beta = 2., double dance = 5., double ddamp = 0.8,
			double fl = 1., double fldamp = 0.99, double gmin = 0.8,
			double gmax = 0.8, double vdamp = 0.1, double sigma=0.1, double pmutdim = 0.01,
			double pmutnp = 0.05, double l = 0.95, bool pgb = false);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void updateMaleVelocity(mayfly &fly);

	void updateFemaleVelocity(mayfly &female, mayfly &male);

	void updatePosition(mayfly &fly);

	void crossover(mayfly &par1, mayfly &par2, mayfly &off1, mayfly &off2);

	bool mutation(mayfly &par, mayfly &off);
};

#endif /* MULTIVARIATE_EVOL_MAYFLY_H_ */
