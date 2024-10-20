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

 [1] Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: Self-adaptive differential
 evolution with fast and reliable convergence performance." 2007 IEEE congress
 on evolutionary computation. IEEE, 2007.

 [2] Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: adaptive differential
 evolution with optional external archive." IEEE Transactions on evolutionary
 computation 13.5 (2009): 945-958.

 [3] Li, Jie, et al. "Power mean based crossover rate adaptive differential
 evolution." International Conference on Artificial Intelligence and Computational
 Intelligence. Springer, Berlin, Heidelberg, 2011.

 [4] Gong, Wenyin, Zhihua Cai, and Yang Wang. "Repairing the crossover rate
 in adaptive differential evolution." Applied Soft Computing 15 (2014): 149-168.
 */

#ifndef MULTIVARIATE_DE_JADE_H_
#define MULTIVARIATE_DE_JADE_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class JadeSearch: public MultivariateOptimizer {

protected:
	bool _archive, _repaircr;
	int _n, _np, _fev, _mfev;
	double _mucr, _muf, _pelite, _c, _tol, _sigma;
	multivariate_problem _f;
	std::vector<double> _lower, _upper, _work, _scr, _sf;
	std::vector<std::vector<double>> _arch;
	std::vector<point> _swarm;

public:
	std::normal_distribution<> _Z { 0., 1. };

	JadeSearch(int mfev, int np, double tol, bool archive = true,
			bool repaircr = true, double pelite = 0.05, double cdamp = 0.1,
			double sigma = 0.07);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();

	double mutate(double *x, double *best, double *xr1, double *xr2,
			double *out, int n, double F, double CR);

	double meanPow(double *values, int n, int p);

	double std(double *values, int n);

	double sampleCauchy();
};

#endif /* MULTIVARIATE_DE_JADE_H_ */
