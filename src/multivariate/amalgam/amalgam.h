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

#ifndef MULTIVARIATE_AMALGAM_H_
#define MULTIVARIATE_AMALGAM_H_

#include <memory>
#include <random>

#include "../../tabular.hpp"

#include "../multivariate.h"

class Amalgam: public MultivariateOptimizer {

protected:
	bool _iamalgam, _noparam, _print;
	int _n, _fev, _mfev, _s, _runs, _nbase, _budget, _nelite, _nams, _nis,
			_nismax, _np, _ss, _t;
	double _tol, _stol, _mincmult, _tau, _alphaams, _deltaams, _etasigma,
			_etashift, _etadec, _etainc, _cmult, _thetasdr, _fbest, _fbestrun,
			_fbestrunold;
	multivariate_problem _f;
	Tabular _table;
	std::vector<double> _lower, _upper, _mu, _muold, _mushift, _mushiftold,
			_best, _tmp, _xavg;
	std::vector<std::vector<double>> _cov, _chol;
	std::vector<point> _sols;

public:
	std::normal_distribution<> _Z { 0., 1. };

	Amalgam(int mfev, double tol, double stol, int np = 0,
			bool iamalgam = true, bool noparam = false, bool print = false);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void runParallel();

	bool converged();

	double computeSDR();

	void updateDistribution();

	int samplePopulation();

	int dchdcm();

};

#endif /* MULTIVARIATE_AMALGAM_H_ */
