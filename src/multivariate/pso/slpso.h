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

 [1] Li, Changhe, Shengxiang Yang, and Trung Thanh Nguyen. "A self-learning particle
 swarm optimizer for global optimization problems." IEEE Transactions on Systems,
 Man, and Cybernetics, Part B (Cybernetics) 42.3 (2011): 627-646.
 */

#ifndef MULTIVARIATE_PSO_SLPSO_H_
#define MULTIVARIATE_PSO_SLPSO_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class SLPSOSearch: public MultivariateOptimizer {

	struct slpso_particle {

		std::vector<double> _x, _v, _pb;
		double _fx, _fpb, _fp;
		std::vector<double> _p, _s;
		std::vector<int> _g, _G;
		int _m;
		bool _CF, _PF;
		double _Pl, _Uf;
	};

protected:
	int _n, _np, _nstrat, _fev, _mfev, _mfes;
	double _omega, _omegamin, _omegamax, _eta, _gamma, _vmax, _tol, _alpha,
			_fabest, _Ufmax;
	multivariate_problem _f;
	std::vector<int> _perm;
	std::vector<double> _lower, _upper, _abest, _r, _vdavg;
	std::vector<slpso_particle> _swarm;

public:
	std::normal_distribution<> _Z { 0., 1. };

	SLPSOSearch(int mfev, double stol, int np, double omegamin = 0.4,
			double omegamax = 0.9, double eta = 1.496, double gamma = 0.01,
			double vmax = 0.2, double Ufmax = 10.);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();

	int roulette(slpso_particle &k);

	// particle updating routines
	void initializeParticle(int index);

	slpso_particle& update(int i, int p, slpso_particle &k);

	double handleBounds(double x, double v, int d);

	void updateAbest(slpso_particle &k);

	// parameter updating routines
	void updatePar();

	void updateParParticle(int index);

	void updateLearningOpt(slpso_particle &k);

	void updateSelectionRatios(slpso_particle &k);
};

#endif /* MULTIVARIATE_PSO_SLPSO_H_ */
