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

 [1] Cheng, Ran & Jin, Yaochu. (2014). A Competitive Swarm Optimizer for Large
 Scale Optimization. IEEE transactions on cybernetics. 45.
 10.1109/TCYB.2014.2322602.

 [2] Mohapatra, Prabhujit, Kedar Nath Das, and Santanu Roy. "A modified competitive
 swarm optimizer for large scale optimization problems." Applied Soft Computing 59
 (2017): 340-362.

 [3] Mohapatra, Prabhujit, Kedar Nath Das, and Santanu Roy. "Inherited competitive
 swarm optimizer for large-scale optimization problems." Harmony Search and Nature
 Inspired Optimization Algorithms. Springer, Singapore, 2019. 85-95.
 */

#ifndef MULTIVARIATE_PSO_CSO_H_
#define MULTIVARIATE_PSO_CSO_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class CSOSearch: public MultivariateOptimizer {

	struct cso_particle {

		std::vector<double> _x, _v, _mean;
		double _f;
		cso_particle *_left, *_right;

		static bool compare(const cso_particle &a, const cso_particle &b) {
			return a._f < b._f;
		}
	};

protected:
	bool _ring, _correct;
	int _n, _fev, _np, _ngroup, _mfev, _pcompete;
	double _vmax, _stol;
	multivariate_problem _f;
	cso_particle *_best;
	std::vector<double> _lower, _upper, _mean, _meanw, _phil, _phih;
	std::vector<cso_particle> _swarm;

public:
	CSOSearch(int mfev, double stol, int np, int pcompete = 3, bool ring = false,
			bool correct = true, double vmax = 0.2);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();

	void computePhi(int m);

	void compete(int istart);

	void updateWinners();
};

#endif /* MULTIVARIATE_PSO_CSO_H_ */
