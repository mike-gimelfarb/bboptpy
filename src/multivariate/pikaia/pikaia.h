/*
 * The original version of this PIKAIA software in fortran 77 is public domain software
 * written by the High Altitude Observatory and available here:
 * https://www.hao.ucar.edu/modeling/pikaia/pikaia.php#sec4. Please see site for detailed licenses.
 *
 * Translated to C++11 by Mike Gimelfarb
 */

#ifndef MULTIVARIATE_PIKAIA_H_
#define MULTIVARIATE_PIKAIA_H_

#include "../multivariate.h"

class PikaiaSearch: public MultivariateOptimizer {

protected:
	int _n, _np, _nd, _ngen, _imut, _irep, _ielite, _ig, _ip1, _ip2, _new,
			_newtot, _fev;
	double _pcross, _pmut, _pmutmn, _pmutmx, _fdif;
	multivariate_problem _f;
	std::vector<int> _gn1, _gn2, _ifit, _jfit;
	std::vector<double> _lower, _upper, _ph, _oldph, _newph, _fitns;

public:
	PikaiaSearch(int np, int ngen, int nd, double pcross = 0.85, int imut = 2,
			double pmut = 0.005, double pmutmn = 0.0005, double pmutmx = 0.25,
			double fdif = 1., int irep = 1, int ielite = 0);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution solution();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	double* row_ptr(int row, int n, double *d);

	void rqsort(int n, double *a, int *p);

	void encode(int n, int nd, double *ph, int *gn);

	void decode(int n, int nd, int *gn, double *ph);

	void cross(int n, int nd, double pcross, int *gn1, int *gn2);

	void mutate(int n, int nd, double pmut, int *gn, int imut);

	void adjmut(int ndim, int n, int np, double *oldph, double *fitns,
			int *ifit, double pmutmn, double pmutmx, double &pmut, int imut);

	void select(int np, int *jfit, double fdif, int &idad);

	void rnkpop(int n, double *arrin, int *indx, int *rank);

	void genrep(int ndim, int n, int np, int ip, double *ph, double *newph);

	void stdrep(const multivariate &ff, int ndim, int n, int np, int irep,
			int ielite, double *ph, double *oldph, double *fitns, int *ifit,
			int *jfit, int &nnew);

	void newpop(const multivariate &ff, int ielite, int ndim, int n, int np,
			double *oldph, double *newph, int *ifit, int *jfit, double *fitns,
			int &nnew);
};

#endif /* MULTIVARIATE_PIKAIA_H_ */
