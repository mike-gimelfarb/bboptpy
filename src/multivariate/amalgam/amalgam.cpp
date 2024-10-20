/*
 Please note, even though the main code for AMALGAM can be available under MIT
 licensed, the dchdcm subroutine is a derivative of LINPACK code that is licensed
 under the 3-Clause BSD license. The other subroutines:

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

 [1] Bosman, Peter AN, J�rn Grahl, and Dirk Thierens. "AMaLGaM IDEAs in
 noiseless black-box optimization benchmarking." Proceedings of the 11th
 Annual Conference Companion on Genetic and Evolutionary Computation
 Conference: Late Breaking Papers. ACM, 2009.
 */

#include "../../blas.h"
#include "../../random.hpp"

#include "amalgam.h"

using Random = effolkronium::random_static;

Amalgam::Amalgam(int mfev, double tol, double stol, int np, bool iamalgam,
		bool noparam, bool print) {
	_tol = tol;
	_stol = stol;
	_mfev = mfev;
	_np = np;
	_iamalgam = iamalgam;
	_noparam = noparam;
	_print = print;
}

void Amalgam::init(const multivariate_problem &f, const double *guess) {

	// basic constants
	_mincmult = 1e-10;
	_tau = 0.35;
	_nelite = 1;

	// prepare problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [AMALGAM]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(_f._lower, _f._lower + _n);
	_upper = std::vector<double>(_f._upper, _f._upper + _n);
	_table = Tabular();

	// parameter-free version
	if (_noparam) {

		// compute the base population size
		if (_iamalgam) {
			_nbase = static_cast<int>(10. * std::sqrt(1. * _n));
		} else {
			_nbase = static_cast<int>(17. + 3. * std::pow(1. * _n, 1.5));
		}

		// for keeping track of the best global solution
		_fbest = std::numeric_limits<double>::infinity();
		_fbestrun = std::numeric_limits<double>::infinity();
		_fbestrunold = std::numeric_limits<double>::infinity();
		_best = std::vector<double>(_n);

		// initialize counters
		_s = 0;
		_budget = _mfev;
		_fev = 0;

		// print headers
		if (_print) {
			_table.setWidth( { 5, 5, 10, 25, 25, 10 });
			_table.printRow("iter", "runs", "pop", "f*", "best f*", "fev");
		}
		return;
	}

	// initialize population size
	if (_np <= 0) {
		if (_iamalgam) {
			_np = static_cast<int>(10. * std::sqrt(1. * _n));
		} else {
			_np = static_cast<int>(17. + 3. * std::pow(1. * _n, 1.5));
		}
	}

	// prepare parameters to run i-amalgam algorithm
	_fev = 0;
	_t = 0;
	_ss = static_cast<int>(_tau * _np);
	if (_iamalgam) {
		const double es = -1.1 * std::pow(_ss, 1.2) / std::pow(1. * _n, 1.6);
		const double esh = -1.2 * std::pow(_ss, 0.31) / std::sqrt(1. * _n);
		_etasigma = 1. - std::exp(es);
		_etashift = 1. - std::exp(esh);
	} else {
		_etasigma = 1.;
		_etashift = 1.;
	}

	// anticipated mean shift
	_alphaams = (0.5 * _tau * _np) / (_np - _nelite);
	_nams = static_cast<int>(_alphaams * (_np - 1));
	_deltaams = 2.;

	// distribution multipliers
	_nis = 0;
	_nismax = 25 + _n;
	_etadec = 0.9;
	_etainc = 1. / _etadec;
	_thetasdr = 1.;
	_cmult = 1.;

	// initialize the population
	_sols.clear();
	for (int m = 0; m < _np; m++) {
		std::vector<double> x(_n);
		for (int i = 0; i < _n; i++) {
			x[i] = Random::get(_lower[i], _upper[i]);
		}
		const point part { x, f._f(&x[0]) };
		_sols.push_back(std::move(part));
	}
	_fev += _np;
	std::sort(_sols.begin(), _sols.end(), point::compare_fitness);

	// initialize the other arrays
	_mu = std::vector<double>(_n, 0.);
	_muold = std::vector<double>(_n, 0.);
	_mushift = std::vector<double>(_n, 0.);
	_mushiftold = std::vector<double>(_n, 0.);
	_tmp = std::vector<double>(_n, 0.);
	_xavg = std::vector<double>(_n, 0.);
	_cov.clear();
	_cov.resize(_n, std::vector<double>(_n, 0.));
	_chol.clear();
	_chol.resize(_n, std::vector<double>(_n, 0.));

	// estimate mean
	for (int m = 0; m < _ss; m++) {
		dxpym(_n, &(_sols[m]._x)[0], 1, &_mu[0], 1);
	}
	dscalm(_n, 1. / _ss, &_mu[0], 1);

	// estimate covariance
	for (int i = 0; i < _n; i++) {
		for (int m = 0; m < _ss; m++) {
			const double xmmu = _sols[m]._x[i] - _mu[i];
			_cov[i][i] += xmmu * xmmu;
			_cov[i][i] /= _ss;
		}
	}
}

void Amalgam::iterate() {

	// parameter-free version
	if (_noparam) {

		// figure out the population size and the number of parallel runs
		const int floorS = _s >> 1;
		if ((_s & 1) == 0) {
			_np = (1 + floorS) * _nbase;
			_runs = 1 << floorS;
		} else {
			_np = (1 << (1 + floorS)) * _nbase;
			_runs = 1;
		}

		// run algorithms amalgam in parallel
		runParallel();

		// print
		if (_print) {
			_table.printRow(_s, _runs, _np, _fbestrun, _fbest, _fev);
		}

		// increment counters
		_s++;
		return;
	}

	// update mean and variance of the Gaussian
	updateDistribution();

	// re-sample parameters
	const int ibest = samplePopulation();
	_fev += _np;

	// update the rest of the parameters
	if (ibest > 0) {
		_nis = 0;
		if (_cmult < 1.) {
			_cmult = 1.;
		}
		const double SDR = computeSDR();
		if (SDR > _thetasdr) {
			_cmult *= _etainc;
		}
	} else {
		if (_cmult <= 1.) {
			_nis++;
		}
		if (_cmult > 1. || _nis >= _nismax) {
			_cmult *= _etadec;
		}
		if (_cmult < 1. && _nis < _nismax) {
			_cmult = 1.;
		}
	}
	_t++;
}

multivariate_solution Amalgam::solution() {
	std::vector<double> sol;
	if (_noparam) {
		sol = _best;
	} else {
		sol = _sols[0]._x;
	}
	return {sol, _fev, converged()};
}

multivariate_solution Amalgam::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	while (true) {
		iterate();
		if (converged()) {
			return solution();
		}
	}
}

void Amalgam::runParallel() {

	// record best values on this run
	_fbestrunold = _fbestrun;
	_fbestrun = std::numeric_limits<double>::infinity();

	for (int r = 1; r <= _runs; r++) {

		// initialize the optimizer
		Amalgam algr { _budget, _tol, _stol, _np, _iamalgam, false, false };

		// perform the optimization
		const auto &sol = algr.optimize(_f, nullptr);
		_fev += sol._fev;
		_budget -= sol._fev;
		const auto &opt = sol._sol;

		// get the best fitness
		const double fitr = _f._f(&opt[0]);
		_fev++;
		_budget--;

		// update local best solution found on this run
		_fbestrun = std::min(_fbestrun, fitr);

		// update global best solution found
		if (fitr < _fbest) {
			_fbest = fitr;
			std::copy(opt.begin(), opt.end(), _best.begin());
		}
	}
}

bool Amalgam::converged() {

	// check number of evaluations
	if (_fev >= _mfev) {
		return true;
	}

	// parameter-free version
	if (_noparam) {

		// check convergence in objective tolerance
		// TODO: check this condition is satisfactory - if not find a better
		// convergence criterion for multiple restarts
		if (_fbestrun != _fbestrunold
				&& std::fabs(_fbestrun - _fbestrunold) <= _tol) {
			return true;
		}
		return false;
	}

	// check minimum multiplier value
	if (_cmult < _mincmult) {
		return true;
	}

	// compute variance of the population fitness values
	double fmean = 0.;
	for (const auto &sol : _sols) {
		fmean += sol._f / _np;
	}
	double fvar = 0.;
	for (const auto &sol : _sols) {
		fvar += std::pow(sol._f - fmean, 2.) / _np;
	}
	if (fvar <= _stol * _stol) {
		return true;
	}
	return false;
}

double Amalgam::computeSDR() {

	// compute the average of all points better than previous best
	std::fill(_xavg.begin(), _xavg.end(), 0.);
	int count = 0;
	for (int m = 1; m < _np; m++) {
		if (_sols[m]._f < _sols[0]._f) {
			dxpym(_n, &(_sols[m]._x)[0], 1, &_xavg[0], 1);
			count++;
		}
	}
	dscalm(_n, 1. / count, &_xavg[0], 1);

	// subtract mean
	daxpym(_n, -1., &_mu[0], 1, &_xavg[0], 1);

	// compute SDR
	double sdr = 0.;
	for (int i = 0; i < _n; i++) {
		_tmp[i] = _xavg[i];
		_tmp[i] -= std::inner_product(_chol[i].begin(), _chol[i].begin() + i,
				_tmp.begin(), 0.);
		_tmp[i] /= _chol[i][i];
		sdr = std::max(sdr, std::fabs(_tmp[i]));
	}
	return sdr;
}

void Amalgam::updateDistribution() {

	// find the best solutions
	std::sort(_sols.begin(), _sols.end(), point::compare_fitness);

	// save current mu
	std::copy(_mu.begin(), _mu.end(), _muold.begin());

	// compute the mean of the nbest solutions
	std::fill(_mu.begin(), _mu.end(), 0.);
	for (int m = 0; m < _ss; m++) {
		daxpym(_n, 1. / _ss, &(_sols[m]._x)[0], 1, &_mu[0], 1);
	}

	// update covariance matrix
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j <= i; j++) {
			_cov[i][j] *= (1. - _etasigma);
			for (int m = 0; m < _ss; m++) {
				_cov[i][j] += (_etasigma / _ss) * (_sols[m]._x[i] - _mu[i])
						* (_sols[m]._x[j] - _mu[j]);
			}
			_cov[j][i] = _cov[i][j];
		}
	}

	// save old mu shift
	std::copy(_mushift.begin(), _mushift.end(), _mushiftold.begin());

	// compute new shifted mu
	if (_t == 1) {
		for (int i = 0; i < _n; i++) {
			_mushift[i] = _mu[i] - _muold[i];
		}
	} else if (_t > 1) {
		for (int i = 0; i < _n; i++) {
			_mushift[i] *= (1. - _etashift);
			_mushift[i] += _etashift * (_mu[i] - _muold[i]);
		}
	} else {
		std::fill(_mushift.begin(), _mushift.end(), 0.);
	}

	// computes the cholesky factor of the covariance matrix
	for (int i = 0; i < _n; i++) {
		std::copy(_cov[i].begin(), _cov[i].end(), _chol[i].begin());
	}
	dchdcm();
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j < i; j++) {
			_chol[i][j] = _chol[j][i];
			_chol[j][i] = 0.;
		}
	}
	const double cmultsqrt = std::sqrt(_cmult);
	for (int i = 0; i < _n; i++) {
		dscalm(_n, cmultsqrt, &(_chol[i])[0], 1);
	}
}

int Amalgam::samplePopulation() {

	// sample from the estimated normal distribution
	for (auto &sol : _sols) {
		for (int i = 0; i < _n; i++) {
			_tmp[i] = Random::get(_Z);
		}
		for (int i = 0; i < _n; i++) {
			sol._x[i] = _mu[i];
			sol._x[i] += std::inner_product(_chol[i].begin(), _chol[i].end(),
					_tmp.begin(), 0.);
		}
	}

	// perturb n_ams random solutions
	// shift the solutions by a multiple of mu_shift
	Random::shuffle(_sols.begin() + 1, _sols.end());
	for (int mm = 1; mm <= _nams; mm++) {
		daxpym(_n, _deltaams * _cmult, &_mushift[0], 1, &(_sols[mm]._x)[0], 1);
	}

	// perform the fitness evaluation
	// find an element that has a better fitness than the best
	int ibest = 0;
	int m = 0;
	for (auto &sol : _sols) {
		if (m > 0) {
			sol._f = _f._f(&(sol._x)[0]);
			if (sol._f < _sols[0]._f) {
				ibest = m;
			}
		}
		m++;
	}
	return ibest;
}

int Amalgam::dchdcm() {

	// internal variables
	int pu, pl, ii, j, k, km1, kp1, l, maxl;
	double temp, maxdia;

	pl = 1;
	pu = 0;
	int info = _n;

	for (k = 1; k <= _n; k++) {

		// reduction loop
		maxdia = _chol[k - 1][k - 1];
		kp1 = k + 1;
		maxl = k;

		// determine the pivot element
		if (k >= pl && k < pu) {
			for (l = kp1; l <= pu; l++) {
				if (_chol[l - 1][l - 1] > maxdia) {
					maxdia = _chol[l - 1][l - 1];
					maxl = l;
				}
			}
		}

		// quit if the pivot element is not positive
		if (maxdia <= 0.) {
			return k - 1;
		}

		// start the pivoting and update jpvt
		if (k != maxl) {
			km1 = k - 1;
			for (ii = 1; ii <= km1; ii++) {
				temp = _chol[ii - 1][k - 1];
				_chol[ii - 1][k - 1] = _chol[ii - 1][maxl - 1];
				_chol[ii - 1][maxl - 1] = temp;
			}
			_chol[maxl - 1][maxl - 1] = _chol[k - 1][k - 1];
			_chol[k - 1][k - 1] = maxdia;
		}

		// reduction step. pivoting is contained across the rows
		_tmp[k - 1] = std::sqrt(_chol[k - 1][k - 1]);
		_chol[k - 1][k - 1] = _tmp[k - 1];
		if (_n >= kp1) {
			for (j = kp1; j <= _n; j++) {
				if (k != maxl) {
					if (j >= maxl) {
						if (j != maxl) {
							temp = _chol[k - 1][j - 1];
							_chol[k - 1][j - 1] = _chol[maxl - 1][j - 1];
							_chol[maxl - 1][j - 1] = temp;
						}
					} else {
						temp = _chol[k - 1][j - 1];
						_chol[k - 1][j - 1] = _chol[j - 1][maxl - 1];
						_chol[j - 1][maxl - 1] = temp;
					}
				}
				_chol[k - 1][j - 1] /= _tmp[k - 1];
				_tmp[j - 1] = _chol[k - 1][j - 1];
				temp = -_chol[k - 1][j - 1];
				for (ii = 1; ii <= j - k; ii++) {
					_chol[kp1 - 1 + ii - 1][j - 1] += temp
							* _tmp[kp1 - 1 + ii - 1];
				}
			}
		}
	}
	return info;
}
