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

 [1] Lasdon, Leon, et al. "Adaptive memory programming for constrained global
 optimization." Computers & Operations Research 37.8 (2010): 1500-1509.

 [2] The AMPGO Solver — AMPGO 0.1.0 documentation. (n.d.). Infinity77.Net.
 Retrieved August 15, 2021, from http://infinity77.net/global_optimization/ampgo.html
 */

#ifndef MULTIVARIATE_AMPTTM_H_
#define MULTIVARIATE_AMPTTM_H_

#include "../../tabular.hpp"

#include "../multivariate.h"

class AMPTTM: public MultivariateOptimizer {

public:
	enum tabu_removal_strategy {
		oldest, farthest
	};

protected:
	bool _print, _improve, _bestxfeas;
	unsigned int _tabutenure;
	int _n, _fev, _gev, _mit, _mfev, _tit, _it;
	double _eps1, _eps2, _bestf;
	tabu_removal_strategy _remove;
	multivariate_problem _f;
	MultivariateOptimizer *_local;
	Tabular _table;
	std::vector<double> _lower, _upper, _guess, _temp, _x0, _s, _bestx;
	std::vector<std::vector<double>> _tabu;

public:
	AMPTTM(MultivariateOptimizer *local, int mfev, bool print = false,
			double eps1 = 0.02, double eps2 = 0.1, int totaliter = 9999,
			int maxiter = 5, unsigned int tabutenure = 5,
			tabu_removal_strategy remove = farthest);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void updateTabu(const double *x);

	multivariate_solution solveLocalProblem(const double *guess);

	multivariate_solution solveProjectionProblem(const double *guess);

	multivariate_solution solveTunnelingProblem(const double *guess);
};

#endif /* MULTIVARIATE_AMPTTM_H_ */
