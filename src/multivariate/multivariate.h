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
 */

#ifndef MULTIVARIATE_H_
#define MULTIVARIATE_H_

#include <memory>
#include <vector>
#include <functional>

#include "../string_utils.h"

typedef std::function<double(const double*)> multivariate;

typedef std::function<void(const double*, double*)> constraints;

typedef std::function<bool(const double*)> blackbox_constraints;

struct multivariate_problem {

	// objective
	multivariate _f;
	int _n;

	// bound constraints
	double *_lower;
	double *_upper;

	// equality + inequality constraints g(x) = 0, h(x) <= 0
	bool _hasc;
	constraints _c;
	int _neq, _nineq;

	// black-box constraints
	bool _hasbbc;
	blackbox_constraints _bbc;

	multivariate_problem() :
			_f( { }), _n(0), _lower(nullptr), _upper(nullptr), _hasc(false), _c(
					{ }), _neq(0), _nineq(0), _hasbbc(false), _bbc( { }) {
	}

	multivariate_problem(const multivariate f, const int n, double *lower,
			double *upper) :
			_f(f), _n(n), _lower(lower), _upper(upper), _hasc(false), _c( { }), _neq(
					0), _nineq(0), _hasbbc(false), _bbc( { }) {
	}

	multivariate_problem(const multivariate f, const int n, double *lower,
			double *upper, const constraints c, const int neq, const int nineq) :
			_f(f), _n(n), _lower(lower), _upper(upper), _hasc(true), _c(c), _neq(
					neq), _nineq(nineq), _hasbbc(false), _bbc( { }) {
	}

	multivariate_problem(const multivariate f, const int n, double *lower,
			double *upper, const blackbox_constraints bbc) :
			_f(f), _n(n), _lower(lower), _upper(upper), _hasc(false), _c( { }), _neq(
					0), _nineq(0), _hasbbc(true), _bbc(bbc) {
	}
};

struct multivariate_solution {

	const std::vector<double> _sol;
	const int _fev, _cev, _bbev;
	const bool _converged;

	multivariate_solution(const std::vector<double> &sol, const int fev,
			const bool converged) :
			_sol(sol), _fev(fev), _cev(0), _bbev(0), _converged(converged) {
	}

	multivariate_solution(const std::vector<double> &sol, const int fev,
			const int cev, const int bbev, const bool converged) :
			_sol(sol), _fev(fev), _cev(cev), _bbev(bbev), _converged(converged) {
	}

	std::string toString() {
		std::string solstr = "";
		for (auto i = _sol.begin(); i != _sol.end(); i++) {
			solstr += std::to_string(*i) + " ";
		}
		std::string result = "";
		result += "x*: " + solstr + "\n";
		result += "objective calls: " + std::to_string(_fev) + "\n";
		result += "constraint calls: " + std::to_string(_cev) + "\n";
		result += "B/B constraint calls: " + std::to_string(_bbev) + "\n";
		result += "converged: ";
		if (_converged) {
			result += "yes";
		} else {
			result += "no/unknown";
		}
		return result;
	}
};

struct point {

	std::vector<double> _x;
	double _f;

	static bool compare_fitness(const point &x, const point &y) {
		return x._f < y._f;
	}

	static bool compare_fitness_ptr(const std::shared_ptr<point> &x,
			const std::shared_ptr<point> &y) {
		return x->_f < y->_f;
	}
};

class MultivariateOptimizer {

public:
	virtual ~MultivariateOptimizer() {
	}

	virtual void init(const multivariate_problem &f, const double *guess) = 0;

	virtual void iterate() = 0;

	virtual multivariate_solution solution() = 0;

	virtual multivariate_solution optimize(const multivariate_problem &f,
			const double *guess) = 0;
};

#endif
