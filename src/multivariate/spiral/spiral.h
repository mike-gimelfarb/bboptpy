/*
 Copyright (c) 2025 Mike Gimelfarb

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

  [1] Tamura, Kenichi, and Keiichiro Yasuda. "Spiral dynamics inspired optimization." 
 Journal of Advanced Computational Intelligence and Intelligent Informatics 15.8 (2011): 
 1116-1122.
 
 [2] Yüzgeç, Uğur, and Tufan İnaç. "Adaptive spiral optimization algorithm for benchmark 
 problems." Bilecik Şeyh Edebali Üniversitesi Fen Bilimleri Dergisi 3.1 (2016): 8-15.

 */

#ifndef MULTIVARIATE_SPIRAL_H_
#define MULTIVARIATE_SPIRAL_H_

#include "../multivariate.h"

class SpiralSearch: public MultivariateOptimizer {

protected:
	int _n, _fev, _mfev, _m, _ibest;
	double _tol, _r, _theta, _taur, _tautheta, _rlow, _rhigh, _thetalow, _thetahigh;
	multivariate_problem _f;
	std::vector<double> _lower, _upper, _fs, _temp, _temp2, _xbest, _rs, _thetas;
	std::vector<std::vector<double>> _points;

public:
    SpiralSearch(int mfev, double tol, const int np, 
		double r=0.95, double theta=std::acos(-1) / 2, 
		double taur=0.0, double tautheta=0.1,
		double rlow=0.9, double rhigh=1.0, double thetalow=0.0, 
		double thetahigh=2 * std::acos(-1));

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void rotate(double *x, int i, int j, double costh, double sinth);

	void rotate_n(double *x, int n, double costh, double sinth);
};

#endif /* MULTIVARIATE_SPIRAL_H_ */
