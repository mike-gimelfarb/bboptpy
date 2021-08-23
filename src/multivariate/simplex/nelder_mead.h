/*
 Original FORTRAN77 version by R ONeill.
 FORTRAN90 version by John Burkardt.
 C++ version by Mike Gimelfarb with changes:
 1. added adaptive parameters
 2. random starts
 3. other initialization schemes

 This code is distributed under the GNU LGPL license.

 ================================================================
 REFERENCES:

 Nelder, John A.; R. Mead (1965). "A simplex method for function minimization".
 Computer Journal. 7 (4): 308�313. doi:10.1093/comjnl/7.4.308

 O'Neill, R. (1971). Algorithm AS 47: Function Minimization Using a Simplex Procedure.
 Journal of the Royal Statistical Society. Series C (Applied Statistics), 20(3), 338-345.
 doi:10.2307/2346772

 Gao, Fuchang & Han, Lixing. (2012). Implementing the Nelder-Mead simplex algorithm
 with adaptive parameters. Computational Optimization and Applications. 51. 259-277.
 10.1007/s10589-010-9329-3.
 */
#ifndef MULTIVARIATE_NELDER_MEAD_H_
#define MULTIVARIATE_NELDER_MEAD_H_

#include "../multivariate.h"

class NelderMead: public MultivariateOptimizer {

public:
	enum simplex_initializer {
		original, spendley, pfeffer, random
	};

protected:
	bool _adapt, _conv;
	int _n, _checkev, _mfev, _ihi, _ilo, _jcount, _icount;
	double _tol, _eps, _rad, _ccoef, _ecoef, _rcoef, _scoef, _del, _rq, _y2star,
			_ylo, _ystar, _ynl;
	multivariate_problem _f;
	simplex_initializer _minit;
	std::vector<double> _start, _lower, _upper, _p2star, _pbar, _pstar, _y,
			_xmin, _step;
	std::vector<std::vector<double>> _p;

public:
	NelderMead(int mfev, double tol, double rad0, int checkev = 10,
			simplex_initializer minit = spendley, bool adapt = true);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	int nelmin();

	void initSimplex();
};

#endif /* MULTIVARIATE_NELDER_MEAD_H_ */