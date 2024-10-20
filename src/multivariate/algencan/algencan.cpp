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

 [1] Birgin, Ernesto G., and José Mario Martínez. "Improving ultimate convergence
 of an augmented Lagrangian method." Optimization Methods and Software 23.2 (2008):
 177-195.
 */

#include <cmath>
#include <numeric>
#include <stdexcept>

#include "algencan.h"

Algencan::Algencan(MultivariateOptimizer *local, int mit, double tol,
		bool print, double tau, double gamma, double lambda0, double mu0) {
	_local = local;
	_mit = mit;
	_tol = tol;
	_print = print;
	_tau = tau;
	_gamma = gamma;
	_lambda0 = lambda0;
	_mu0 = mu0;
}

void Algencan::init(const multivariate_problem &f, const double *guess) {

	// define problem
	if (!f._hasc || (f._neq <= 0 && f._nineq <= 0)) {
		throw std::invalid_argument(
				"Error [ALGENCAN]: problem does not have (in)equality constraints.");
	}
	_f = f;
	_n = f._n;
	_m = f._neq;
	_p = f._nineq;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// define memory
	_ghx = std::vector<double>(_m + _p);
	_x0 = std::vector<double>(guess, guess + _n);
	_localconv = false;
	_table = Tabular();

	// other parameters
	_lambdamin = -1e20;
	_lambdamax = +1e20;
	_mumax = +1e20;
	_it = 1;
	_fev = _cev = _bbev = 0;
	_mu = std::vector<double>(_p, _mu0);
	_lambda = std::vector<double>(_m, _lambda0);
	_xbest = std::vector<double>(guess, guess + _n);
	_icmbest = std::numeric_limits<double>::infinity();

	// define initial trust region
	_rho = initialRho(&_x0[0]);
}

void Algencan::iterate() {
	solveLocal();
	updateRho();
	updateMultipliers();
	_it++;

	// print progress
	const double xicm = icm();
	if (_print) {
		const double lag = lagrangian(&_x0[0]);
		_table.printRow(_it - 1, _f._f(&_x0[0]), xicm, lag, _localconv,
				_fev, _rho);
	}

	// update most feasible point solution found
	if (xicm < _icmbest) {
		_icmbest = xicm;
		std::copy(_x0.begin(), _x0.end(), _xbest.begin());
	}
}

multivariate_solution Algencan::solution(){
	return {_xbest, _fev, _cev, _bbev, icm() <= _tol};
}

multivariate_solution Algencan::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);

	// print header and initialization
	if (_print) {
		_table.setWidth( { 5, 25, 25, 25, 5, 10, 25 });
		_table.printRow("iter", "f*", "icm", "L", "conv", "fev", "rho");
		const double L = lagrangian(&_x0[0]);
		_table.printRow(_it - 1, _f._f(&_x0[0]), icm(), L, false, _fev, _rho);
	}

	// main loop
	bool conv = false;
	while (_it < _mit) {
		iterate();

		// check convergence
		if (icm() <= _tol) {
			conv = true;
			break;
		}
	}
	return {_xbest, _fev, _cev, _bbev, conv};
}

double Algencan::lagrangian(const double *x) {

	// evaluate function
	const double fx = _f._f(x);
	_fev++;

	// evaluate constraints
	_f._c(x, &_ghx[0]);
	_cev++;

	// compute barrier
	double hpen = 0.;
	for (int i = 0; i < _m; i++) {
		const double slack = _ghx[i] + _lambda[i] / _rho;
		hpen += std::pow(slack, 2.);
	}
	hpen *= (_rho / 2.);
	double gpen = 0.;
	for (int i = 0; i < _p; i++) {
		const double slack = _ghx[i + _m] + _mu[i] / _rho;
		gpen += std::pow(std::max(0., slack), 2.);
	}
	gpen *= (_rho / 2.);
	return fx + hpen + gpen;
}

double Algencan::initialRho(const double *x) {

	// evaluate function
	const double fx = _f._f(x);
	_fev++;

	// evaluate constraints
	_f._c(x, &_ghx[0]);
	_cev++;

	const double hnorm2 = std::inner_product(_ghx.begin(), _ghx.begin() + _m,
			_ghx.begin(), 0.);
	for (int i = 0; i < _p; i++) {
		_ghx[i + _m] = std::max(0., _ghx[i + _m]);
	}
	const double gnorm2 = std::inner_product(_ghx.begin() + _m, _ghx.end(),
			_ghx.begin() + _m, 0.);

	// evaluate initial penalty using equation 7
	double rho0;
	if (hnorm2 + gnorm2 > 0.) {
		rho0 = 2. * std::fabs(fx) / (hnorm2 + gnorm2);
	} else {
		rho0 = 10.;
	}
	return std::max(1e-6, std::min(rho0, 10.));
}

void Algencan::solveLocal() {

	// define the objective
	const multivariate &L = std::bind(&Algencan::lagrangian, this,
			std::placeholders::_1);
	multivariate_problem lag;
	if (_f._hasbbc) {
		lag = multivariate_problem { L, _n, &_lower[0], &_upper[0], _f._bbc };
	} else {
		lag = multivariate_problem { L, _n, &_lower[0], &_upper[0] };
	}

	// optimize
	const auto &sol = _local->optimize(lag, &_x0[0]);
	_x0 = sol._sol;

	// evaluate constraints at new solution
	_f._c(&_x0[0], &_ghx[0]);
	_cev++;
	_bbev += sol._bbev;
	_localconv = sol._converged;
}

void Algencan::updateRho() {

	// equation 4
	double vnorm = 0.;
	for (int i = 0; i < _p; i++) {
		const double v = std::max(_ghx[i + _m], -_mu[i] / _rho);
		const double vabs = std::abs(v);
		vnorm = std::max(vnorm, vabs);
	}

	// compute the norm of h
	double hnorm = 0.;
	for (int i = 0; i < _m; i++) {
		const double habs = std::abs(_ghx[i]);
		hnorm = std::max(hnorm, habs);
	}

	// increase the penalty when feasibility does not improve
	_rhoold = _rho;
	if (_it != 1
			&& std::max(hnorm, vnorm) > _tau * std::max(_holdnorm, _voldnorm)) {
		_rho *= _gamma;
	}
	_holdnorm = hnorm;
	_voldnorm = vnorm;
}

void Algencan::updateMultipliers() {

	// safeguarded estimate of lambda
	for (int i = 0; i < _m; i++) {
		const double step = _lambda[i] + _rhoold * _ghx[i];
		_lambda[i] = std::max(_lambdamin, std::min(step, _lambdamax));
	}

	// safeguarded estimate of mu
	for (int i = 0; i < _p; i++) {
		const double step = _mu[i] + _rhoold * _ghx[i + _m];
		_mu[i] = std::max(0., std::min(step, _mumax));
	}
}

double Algencan::icm() {
	return std::max(_holdnorm, _voldnorm);
}
