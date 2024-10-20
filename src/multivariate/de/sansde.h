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

 [1] Yang, Zhenyu, Ke Tang, and Xin Yao. "Self-adaptive differential evolution
 with neighborhood search." 2008 IEEE congress on evolutionary computation
 (IEEE World Congress on Computational Intelligence). IEEE, 2008.

 [2] Gong, Wenyin, Zhihua Cai, and Yang Wang. "Repairing the crossover rate in
 adaptive differential evolution." Applied Soft Computing 15 (2014): 149-168.
 */

#ifndef MULTIVARIATE_DE_SANSDE_H_
#define MULTIVARIATE_DE_SANSDE_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class SaNSDESearch: public MultivariateOptimizer {

	struct sansde_point {

		std::vector<double> _x;
		double _f, _cr;

		static bool compare_fitness(const sansde_point &x,
				const sansde_point &y) {
			return x._f < y._f;
		}
	};

protected:
	bool _repaircr;
	int _n, _np, _it, _fev, _mfev, _ncrref, _npup, _ncrup;
	double _tol, _p, _fp, _crrec, _crdeltaf, _crm;
	multivariate_problem _f;
	std::vector<int> _pns, _pnf;
	std::vector<double> _lower, _upper, _work, _fpns, _fpnf;
	std::vector<sansde_point> _swarm;

public:
	std::normal_distribution<> _Z { 0., 1. };

	SaNSDESearch(int mfev, int np, double tol, bool repaircr = true, int crref =
			5, int pupdate = 50, int crupdate = 25);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();

	double mutate(double *x, double *best, double *xr1, double *xr2,
			double *xr3, double *out, int n, double F, double CR, int strat);

	double sampleCauchy();
};

#endif /* MULTIVARIATE_DE_SANSDE_H_ */
