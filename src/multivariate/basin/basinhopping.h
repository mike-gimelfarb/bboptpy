/*
 Copyright © 2003-2019 SciPy Developers.
 Copyright 2021 Mike Gimelfarb for C++ version
 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification, are
 permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

 Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 Neither the name of Enthought nor the names of the SciPy Developers may be
 used to endorse or promote products derived from this software without specific
 prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ================================================================
 REFERENCES:

 [1] Wales, David J.; Doye, Jonathan P. K. (1997-07-10). "Global Optimization by Basin-Hopping
 and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms".
 The Journal of Physical Chemistry A. 101 (28): 5111–5116. arXiv:cond-mat/9803344
 */

#ifndef MULTIVARIATE_BASINHOPPING_H_
#define MULTIVARIATE_BASINHOPPING_H_

#include <memory>

#include "../../tabular.hpp"

#include "../multivariate.h"

struct StepsizeStrategy {

public:
	double _stepsize;

	StepsizeStrategy(double stepsize);

	virtual ~StepsizeStrategy() {
	}

	virtual void takeStep(int n, double *x, double *lower, double *upper);

	virtual void update(bool accept);
};

struct AdaptiveStepsizeStrategy: public StepsizeStrategy {

protected:
	int _int, _nstep, _naccept;
	double _acceptp, _fac;

public:
	AdaptiveStepsizeStrategy(double stepsize = 1., double accept_rate = 0.5,
			int interval = 5, double factor = 0.9);

	void takeStep(int n, double *x, double *lower, double *upper);

	void update(bool accept);

private:
	void adjustStepSize();
};

struct MetropolisHastings {

protected:
	double _beta;

public:
	MetropolisHastings(double t);

	virtual ~MetropolisHastings() {
	}

	virtual bool accept(double fnew, double fold);
};

class BasinHopping: public MultivariateOptimizer {

protected:
	bool _print;
	int _n, _it, _mit, _fev, _cev, _bbev;
	double _energy, _temp, _bestenergy;
	multivariate_problem _f;
	MultivariateOptimizer *_minimizer;
	StepsizeStrategy *_stepstrat;
	MetropolisHastings _acceptance;
	Tabular _table;
	std::vector<double> _guess, _lower, _upper, _x, _bestx;

public:
	BasinHopping(MultivariateOptimizer *minimizer, StepsizeStrategy *stepstrat,
			bool print = true, int mit = 99, double temp = 1.);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);
};

#endif /* MULTIVARIATE_BASINHOPPING_H_ */
