/*
 Copyright � 2003-2019 SciPy Developers.
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

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS �AS IS� AND
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
 The Journal of Physical Chemistry A. 101 (28): 5111�5116. arXiv:cond-mat/9803344
 */

#include "../evol/basinhopping.h"

#include <cmath>
#include <iostream>
#include "random.hpp"

using Random = effolkronium::random_static;

StepsizeStrategy::StepsizeStrategy(double stepsize) {
	_stepsize = stepsize;
}

void StepsizeStrategy::takeStep(int n, double *x, double *lower,
		double *upper) {
	for (int i = 0; i < n; i++) {
		x[i] += _stepsize * Random::get(-1., 1.) * (upper[i] - lower[i]);
		const double margin = 0.05 * (upper[i] - lower[i]);
		if (x[i] < lower[i] + margin) {
			x[i] = lower[i] + margin;
		} else if (x[i] > upper[i] - margin) {
			x[i] = upper[i] - margin;
		}
	}
}

void StepsizeStrategy::update(bool accept) {
}

AdaptiveStepsizeStrategy::AdaptiveStepsizeStrategy(double stepsize,
		double accept_rate, int interval, double factor) :
		StepsizeStrategy(_stepsize) {
	_stepsize = stepsize;
	_acceptp = accept_rate;
	_int = interval;
	_fac = factor;
	_nstep = _naccept = 0;
}

void AdaptiveStepsizeStrategy::takeStep(int n, double *x, double *lower,
		double *upper) {
	_nstep++;
	if (_nstep % _int == 0) {
		adjustStepSize();
	}
	StepsizeStrategy::takeStep(n, x, lower, upper);
}

void AdaptiveStepsizeStrategy::adjustStepSize() {
	const double acceptp = (1. * _naccept) / _nstep;
	if (acceptp > _acceptp) {
		_stepsize /= _fac;
	} else {
		_stepsize *= _fac;
	}
}

void AdaptiveStepsizeStrategy::update(bool accept) {
	if (accept) {
		_naccept++;
	}
}

MetropolisHastings::MetropolisHastings(double t) {
	if (t == 0.) {
		_beta = std::numeric_limits<double>::infinity();
	} else {
		_beta = 1. / t;
	}
}

bool MetropolisHastings::accept(double fnew, double fold) {
	const double w = std::exp(std::min(0., -(fnew - fold) * _beta));
	return w >= Random::get(0., 1.);
}

BasinHopping::BasinHopping(MultivariateOptimizer *minimizer, // @suppress("Class members should be properly initialized")
		StepsizeStrategy *stepstrat, int mit, double temp, bool print) :
		_acceptance(1.) {
	_minimizer = minimizer;
	_stepstrat = stepstrat;
	_temp = temp;
	_mit = mit;
	_print = print;
}

void BasinHopping::init(multivariate f, const int n, double *guess,
		double *lower, double *upper) {

	// define the problem
	_n = n;
	_f = f;
	_guess = std::vector<double>(guess, guess + n);
	_lower = std::vector<double>(lower, lower + n);
	_upper = std::vector<double>(upper, upper + n);

	// define other objects
	_acceptance = MetropolisHastings(_temp);
	_it = 0;
	_fev = 0;

	// find an initial solution
	auto sol = _minimizer->optimize(_f, _n, &_guess[0], &_lower[0], &_upper[0]);
	_x = std::vector<double>(sol._sol);
	_energy = _f(&sol._sol[0]);
	_fev += sol._fev + 1;

	// set best solution
	_bestx = std::vector<double>(_x);
	_bestenergy = _energy;

	// print
	if (_print) {
		std::cout << "it" << "\t" << "f" << "\t" << "conv" << "\t" << "step"
				<< "\t" << "accept" << "\t" << "f*" << "\t" << "fev"
				<< std::endl;
		std::cout << "-1" << "\t" << _energy << "\t" << sol._converged << "\t"
				<< _stepstrat->_stepsize << "\t" << true << "\t" << _bestenergy
				<< "\t" << _fev << std::endl;
	}
}

void BasinHopping::iterate() {

	// take a random step
	std::vector<double> x1(_x);
	_stepstrat->takeStep(_n, &x1[0], &_lower[0], &_upper[0]);

	// do the next local minimization
	auto sol = _minimizer->optimize(_f, _n, &x1[0], &_lower[0], &_upper[0]);
	const double new_energy = _f(&(sol._sol)[0]);
	_fev += sol._fev + 1;

	// decide to accept or reject the move
	const bool accept = _acceptance.accept(new_energy, _energy);
	if (accept) {
		_energy = new_energy;
		std::copy(sol._sol.begin(), sol._sol.end(), _x.begin());
	}

	// update step size parameters
	_stepstrat->update(accept);

	// update best solution
	if (new_energy < _bestenergy) {
		_bestenergy = new_energy;
		std::copy(sol._sol.begin(), sol._sol.end(), _bestx.begin());
	}

	// print
	if (_print) {
		std::cout << _it << "\t" << new_energy << "\t" << sol._converged << "\t"
				<< _stepstrat->_stepsize << "\t" << accept << "\t"
				<< _bestenergy << "\t" << _fev << std::endl;
	}
	_it++;
}

multivariate_solution BasinHopping::optimize(multivariate f, const int n,
		double *guess, double *lower, double *upper) {
	init(f, n, guess, lower, upper);
	while (_it < _mit) {
		iterate();
	}
	return {_bestx, _fev, false};
}
