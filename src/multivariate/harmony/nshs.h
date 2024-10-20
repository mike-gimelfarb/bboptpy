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

 [1] Luo, Kaiping. "A Novel Self-Adaptive Harmony Search Algorithm."
 Journal of Applied Mathematics 2013.1 (2013): 653749.
 */

#ifndef MULTIVARIATE_NSHS_H_
#define MULTIVARIATE_NSHS_H_

#include <memory>

#include "../multivariate.h"

struct harmony {

	std::vector<double> _x;
	double _f;

	static bool compare_fitness(const harmony &x, const harmony &y) {
		return x._f < y._f;
	}
};

class NSHS: public MultivariateOptimizer {

protected:
	int _mfev, _n, _hms, _it, _mit, _fev;
	double _hmcr, _fstd, _fstdmin, _tunerange;
	multivariate_problem _f;
	std::vector<double> _lower, _upper;
	std::vector<harmony> _hm;
	harmony _temp;
	harmony *_best;

public:
	NSHS(int mfev, int hms, double fstdmin = 0.0001);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void generate_harmony();

	void replace();

	void calculate_std();
};

#endif /* MULTIVARIATE_NSHS_H_ */
