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

 [1] Box, M. J. "A new method of constrained optimization and a comparison
 with other methods." The Computer Journal 8.1 (1965): 42-52.

 [2] Guin, J. A. "Modification of the complex method of constrained
 optimization." The Computer Journal 10.4 (1968): 416-417.

 [3] Krus P., Andersson J., Optimizing Optimization for Design Optimization,
 in Proceedings of ASME Design Automation Conference, Chicago, USA, September 2-6, 2003
 */

#ifndef MULTIVARIATE_BOX_H_
#define MULTIVARIATE_BOX_H_

#include "../multivariate.h"

class BoxComplex: public MultivariateOptimizer {

protected:
	bool _movetobest, _adaptalpha;
	int _n, _mfev, _nbox, _fev, _bbev;
	double _xtol, _ftol, _alpha, _rfac, _gamma, _fmin, _fmax, _kf;
	multivariate_problem _f;
	std::vector<double> _lower, _upper, _center, _center0, _xref, _min, _max;
	std::vector<point> _box;

public:
	BoxComplex(int mfev, double ftol, double xtol, double alpha = 0.,
			double rfac = 0., double rforget = 0.3, int nbox = 0,
			bool movetobest = true);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	bool converged();
};

#endif /* MULTIVARIATE_BOX_H_ */
