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

 [1] Hager, W. W. "A derivative-based bracketing scheme for univariate
 minimization and the conjugate gradient method." Computers & Mathematics with
 Applications 18.9 (1989): 779-795.
 */

#ifndef CUBIC_H_
#define CUBIC_H_

#include "deriv/univariate_differentiable.h"

template<typename T> class CubicSearch: public UnivariateDifferentiableOptimizer<
		T> {

protected:
	int _mfev;
	double _rtol, _atol;

public:
	CubicSearch(double rtol, double atol, int mfev);

	T step(T a, T b, T c, T tau);

	T cubic(T a, T fa, T da, T b, T fb, T db);

	void update(T a, T fa, T dfa, T b, T c, T fc, T dfc, T &o1, T &o2);

	solution_for_differentiable<T> optimize(univariate<T> f, univariate<T> df,
			T a, T b);
};

#include "deriv/cubic.tpp"

#endif /* CUBIC_H_ */
