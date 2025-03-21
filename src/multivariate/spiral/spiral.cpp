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

#include "spiral.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include "../../random.hpp"


using Random = effolkronium::random_static;

SpiralSearch::SpiralSearch(int mfev, double tol, int np, double r, double theta,
	double taur, double tautheta, 
	double rlow, double rhigh, double thetalow, double thetahigh) {
	_mfev = mfev;
	_tol = tol;
	_m = np;
	_r = r;
	_theta = theta;

	// for adaptive r and theta
	_taur = taur;
	_tautheta = tautheta;
	_rlow = rlow;
	_rhigh = rhigh;
	_thetalow = thetalow;
	_thetahigh = thetahigh;
}

void SpiralSearch::init(const multivariate_problem &f, const double *guess) {
	
	// prepare problem
	if (f._hasc || f._hasbbc) {
		std::cerr
				<< "Warning [SpiralSearch]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// prepare initial points
	_points.clear();
	_points.resize(_m, std::vector<double>(_n, 0.));
	_fs = std::vector<double>(_m, 0.);
	double fbest = std::numeric_limits<double>::infinity();
	_ibest = 0;
	_xbest = std::vector<double>(_n, 0.);
	for (int i = 0; i < _m; i++){
		for (int d = 0; d < _n; d++){
			_points[i][d] = Random::get(_lower[d], _upper[d]);
		}
		_fs[i] = _f._f(&(_points[i])[0]);
		if (_fs[i] < fbest){
			fbest = _fs[i];
			_ibest = i;
		}
	}
	std::copy(_points[_ibest].begin(), _points[_ibest].end(), _xbest.begin());
	_fev = _m;
	_temp = std::vector<double>(_n, 0.);
	_temp2 = std::vector<double>(_n, 0.);

	// initialize radius and angle
	_rs = std::vector<double>(_m, 0.);
	_thetas = std::vector<double>(_m, 0.);
	for (int i = 0; i < _m; i++){
		_rs[i] = _r;
		_thetas[i] = _theta;
	}
}

void SpiralSearch::iterate() {

	// update the radius and angle
	for (int i = 0; i < _m; i++){
		if (Random::get(0.0, 1.0) < _taur){
			_rs[i] = Random::get(_rlow, _rhigh);
		}
		if (Random::get(0.0, 1.0) < _tautheta){
			_thetas[i] = Random::get(_thetalow, _thetahigh);
		}
	}

	// rotate the points
	for (int i = 0; i < _m; i++){

		// rotate the best point
		const double costh = std::cos(_thetas[i]);
		const double sinth = std::sin(_thetas[i]);
		std::copy(_xbest.begin(), _xbest.end(), _temp.begin());
		rotate_n(&_temp[0], _n, costh, sinth);

		// rotate the current points
		std::copy(_points[i].begin(), _points[i].end(), _temp2.begin());
		rotate_n(&_temp2[0], _n, costh, sinth);
		for (int d = 0; d < _n; d++){
			_points[i][d] = _rs[i] * _temp2[d] - _rs[i] * _temp[d] + _xbest[d];
		}
	}

	// update the best point
	double fbest = std::numeric_limits<double>::infinity();
	_ibest = 0;
	for (int i = 0; i < _m; i++){
		_fs[i] = _f._f(&(_points[i])[0]);
		if (_fs[i] < fbest){
			fbest = _fs[i];
			_ibest = i;
		}
	}
	std::copy(_points[_ibest].begin(), _points[_ibest].end(), _xbest.begin());
	_fev += _m;
}

multivariate_solution SpiralSearch::solution(){
	return {_xbest, _fev, false};
}

multivariate_solution SpiralSearch::optimize(const multivariate_problem &f,
		const double *guess) {
	
	// initialize parameters
	init(f, guess);

	// main iteration loop over generations
	bool converge = false;
	while (_fev < _mfev) {

		// perform a single generation
		iterate();

		// test convergence in standard deviation
		if (false) {
			converge = true;
			break;
		}
	}
	return {_xbest, _fev, converge};
}

void SpiralSearch::rotate(double *x, int i, int j, double costh, double sinth){	
	const double new_xi = costh * x[i - 1] - sinth * x[j - 1];
	const double new_xj = sinth * x[i - 1] + costh * x[j - 1];
	x[i - 1] = new_xi;
	x[j - 1] = new_xj;
}

void SpiralSearch::rotate_n(double *x, int n, double costh, double sinth){
	for (int i = n - 1; i >= 1; i--){
		for (int j = i; j >= 1; j--){
			rotate(x, n - i, n + 1 - j, costh, sinth); 
		}
	}
}
