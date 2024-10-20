/*
 The original Fortran code was written by Richard Brent and made
 available by the Stanford Linear Accelerator Center, dated 3/1/73.
 Since this code contains no copyright statements and is dated prior to
 1977, under US copyright law it is in the public domain (not copyrighted).

 This C++ version is MIT licensed

 ================================================================
 REFERENCES:

 Nelder, John A.; R. Mead (1965). "A simplex method for function minimization".
 Computer Journal. 7 (4): 308–313. doi:10.1093/comjnl/7.4.308

 O'Neill, R. (1971). Algorithm AS 47: Function Minimization Using a Simplex Procedure.
 Journal of the Royal Statistical Society. Series C (Applied Statistics), 20(3), 338-345.
 doi:10.2307/2346772

 Gao, Fuchang & Han, Lixing. (2012). Implementing the Nelder-Mead simplex algorithm
 with adaptive parameters. Computational Optimization and Applications. 51. 259-277.
 10.1007/s10589-010-9329-3.
 */

#ifndef MULTIVARIATE_PRAXIS_H_
#define MULTIVARIATE_PRAXIS_H_

#include "../multivariate.h"

class Praxis: public MultivariateOptimizer {

protected:
	double _tol, _mstep;

public:
	Praxis(double tol, double mstep);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	int praxis(int n, double *x, const multivariate &f);

	double flin(int n, int jsearch, double l, const multivariate &f, double *x,
			int &nf, double *v, double *q0, double *q1, double &qd0,
			double &qd1, double &qa, double &qb, double &qc);

	void minfit(int n, double tol, double *a, double *q);

	void minny(int n, int jsearch, int nits, double &d2, double &x1, double &f1,
			bool fk, const multivariate &f, double *x, double t, double h,
			double *v, double *q0, double *q1, int &nl, int &nf, double dmin,
			double ldt, double &fx, double &qa, double &qb, double &qc,
			double &qd0, double &qd1);

	void quad(int n, const multivariate &f, double *x, double t, double h,
			double *v, double *q0, double *q1, int &nl, int &nf, double dmin,
			double ldt, double &fx, double &qf1, double &qa, double &qb,
			double &qc, double &qd0, double &qd1);

	void tr_mat(int n, double *a);

	void svsort(int n, double *d, double *v);
};

#endif /* MULTIVARIATE_PRAXIS_H_ */
