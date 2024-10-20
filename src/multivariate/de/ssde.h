/*
 Copyright (c) 2024 Mike Gimelfarb

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

 [1] Kumar, A., Misra, R. K., Singh, D., Mishra, S., & Das, S. (2019).
 The spherical search algorithm for bound-constrained global optimization
 problems. Applied Soft Computing, 85, 105734.

 [2] Zhao, J., Zhang, B., Guo, X., Qi, L., & Li, Z. (2022). Self-adapting
 spherical search algorithm with differential evolution for global optimization.
 Mathematics, 10(23), 4519.
 */

#ifndef MULTIVARIATE_DE_SSDE_H_
#define MULTIVARIATE_DE_SSDE_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class SSDESearch: public MultivariateOptimizer {

protected:
	bool _usede, _repaircr;
	int _n, _np, _npmin, _npinit, _mfev, _fev, _patience, _convcount, _H, _k,
			_kCR;
	double _tol, _ptop;
	multivariate_problem _f;
	std::vector<int> _iwork;
	std::vector<double> _lower, _upper, _b, _z, _y, _work, _L1, _L2, _LCR, _SR,
			_SC, _SCR, _w, _wCR;
	std::vector<std::vector<double>> _A;
	std::vector<point> _swarm;

public:
	std::normal_distribution<> _Z { 0., 1. };

	SSDESearch(int mfev, int npinit, double tol, int patience = 1000,
			int npmin = 4, double ptop = 0.11, int h = 100, bool usede = true,
			bool repaircr = true);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();

	void randA();

	void computeTrialPoint(const int i, const double ci);

	double sampleCauchy();

	void sample3(const int i, int &pi, int &qi, int &ri);
};

#endif /* MULTIVARIATE_DE_SSDE_H_ */
