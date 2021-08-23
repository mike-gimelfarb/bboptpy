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

 [1] Wang, Hui, et al. "Randomly attracted firefly algorithm with neighborhood
 search and dynamic parameter adjustment mechanism." Soft Computing 21.18
 (2017): 5325-5339.

 [2] I. Fister Jr., X.-S. Yang, I. Fister, J. Brest, Memetic firefly algorithm
 for combinatorial optimization, in Bioinspired Optimization Methods and their
 Applications (BIOMA 2012), B. Filipic and J.Silc, Eds. Jozef Stefan
 Institute, Ljubljana, Slovenia, 2012.

 [3] Yu S, Zhu S, Ma Y, Mao D. Enhancing firefly algorithm using generalized
 opposition-based learning. Computing. 2015 :97(7) 741�754.

 [4] Shakarami MR, Sedaghati R. A new approach for network reconfiguration
 problem in order to deviation bus voltage minimization with regard to
 probabilistic load model and DGs. International Journal of Electrical,
 Computer, Energetic, Electronic and Communication Engineering.
 2014;8(2):430�5.
 */

#ifndef MULTIVARIATE_FIREFLY_H_
#define MULTIVARIATE_FIREFLY_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class FireflySearch: public MultivariateOptimizer {

public:
	enum noise_type {
		uniform, gauss, cauchy, none
	};

	enum search_strategy {
		geometric, sh2014, memetic
	};

	struct firefly {

		std::vector<double> _x, _xb;
		double _f, _alpha;
	};

protected:
	bool _nsearch, _osearch;
	int _it, _fev, _np, _n, _k, _mfev, _maxit;
	double _mina, _maxa, _wbprob, _bmin, _bmax, _gamma, _alpha0, _decay;
	noise_type _noise;
	search_strategy _strategy;
	multivariate_problem _f;
	firefly *_best;
	std::vector<int> _tmpk;
	std::vector<double> _lower, _upper, _tmpd, _tmp4;
	std::vector<std::vector<double>> _tmpx;
	std::vector<firefly> _swarm;

public:
	std::normal_distribution<> _Z { 0., 1. };

	FireflySearch(int mfev, int np, double gamma, double alpha0, double decay =
			0.97, double bmin = 0.1, double bmax = 0.9,
			search_strategy strategy = sh2014, noise_type noise = uniform,
			bool nsearch = true, int ns = 2, bool osearch = true,
			double wbprob = 0.25);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void updateFireflies();

	double applyStrategy(int i);

	double computeDistance(double *x, double *y);

	void rectifyBounds(double *x);

	void updateStats();

	void sample2FromNeighborhood(int i);

	int sample1FromSwarm(int i);

	void sample2FromSwarm(int i);

	void sample3Uniform();

	double sampleNoise();

	double sampleCauchy();
};

#endif /* MULTIVARIATE_FIREFLY_H_ */