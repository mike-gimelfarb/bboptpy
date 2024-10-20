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

 [1] Zhan, Zhi-Hui, Jun Zhang, Yun Li, and Henry Shu-Hung Chung. "Adaptive
 particle swarm optimization." IEEE Transactions on Systems, Man, and Cybernetics,
 Part B (Cybernetics) 39, no. 6 (2009): 1362-1381.
 */

#ifndef MULTIVARIATE_PSO_APSO_H_
#define MULTIVARIATE_PSO_APSO_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class APSOSearch: public MultivariateOptimizer {

	struct apso_particle {

		std::vector<double> _x, _v, _xb;
		double _f, _fb;
	};

protected:

	const std::vector<std::vector<int>> _rulebase = { //
			{ 1, 1, 1, 1 }, // state is S1, no overlaps
					{ 2, 2, 2, 2 }, // state is S2, no overlaps
					{ 3, 3, 3, 3 }, // state is S3, no overlaps
					{ 4, 4, 4, 4 }, // state is S4, no overlaps
					{ 1, 2, 2, 1 }, // state transition S1 => S2
					{ 2, 2, 3, 3 }, // state transition S2 => S3
					{ 1, 1, 4, 4 }, // state transition S4 => S1
			};

	int _n, _it, _fev, _maxit, _state;
	double _smin, _smax, _w, _c1, _c2, _vdamp, _fbest;
	multivariate_problem _f;
	std::vector<double> _p, _ws, _wmu, _lower, _upper, _xbest;
	std::vector<apso_particle> _swarm;

	bool _correct;
	int _np, _mfev;
	double _tol;

public:
	std::normal_distribution<> _Z { 0., 1. };

	APSOSearch(int mfev, double tol, int np, bool correct = true);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();

	// swarm updates
	void updateSwarm();

	void updateParticle(apso_particle &particle);

	void updateElitist();

	// parameter updates
	void updateParameters();

	void updatec1c2(double f, int state);

	double getf();

	int nextState(double f);

	double mu(double f, int i);
};

#endif /* MULTIVARIATE_PSO_APSO_H_ */
