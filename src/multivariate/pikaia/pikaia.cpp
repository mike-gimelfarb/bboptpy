/*
 * The original version of this PIKAIA software in fortran 77 is public domain software
 * written by the High Altitude Observatory and available here:
 * https://www.hao.ucar.edu/modeling/pikaia/pikaia.php#sec4. Please see site for detailed licenses.
 *
 * Translated to C++11 by Mike Gimelfarb
 */

#include <cmath>
#include <iostream>

#include "../../random.hpp"

#include "pikaia.h"

using Random = effolkronium::random_static;

PikaiaSearch::PikaiaSearch(int np, int ngen, int nd, double pcross, int imut,
		double pmut, double pmutmn, double pmutmx, double fdif, int irep,
		int ielite) {
	_np = np;
	_ngen = ngen;
	_nd = nd;
	_pcross = pcross;
	_imut = imut;
	_pmut = pmut;
	_pmutmn = pmutmn;
	_pmutmx = pmutmx;
	_fdif = fdif;
	_irep = irep;
	_ielite = ielite;
}

void PikaiaSearch::init(const multivariate_problem &f, const double *guess) {

	// initialize problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [PIKAIA]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// initialize memory
	_ph = std::vector<double>(2 * _n);
	_oldph = std::vector<double>(_np * _n);
	_newph = std::vector<double>(_np * _n);
	_gn1 = std::vector<int>(_n * _nd);
	_gn2 = std::vector<int>(_n * _nd);
	_ifit = std::vector<int>(_np);
	_jfit = std::vector<int>(_np);
	_fitns = std::vector<double>(_np);

	// Compute initial (random but bounded) phenotypes
	for (int ip = 1; ip <= _np; ip++) {
		double *oldph_ptr = row_ptr(ip, _n, &_oldph[0]);
		for (int k = 1; k <= _n; k++) {
			oldph_ptr[k - 1] = Random::get(0., 1.);
		}
		_fitns[_np - 1] = _f._f(oldph_ptr);
	}
	_fev = _np;

	// Rank initial population by fitness order
	rnkpop(_np, &_fitns[0], &_ifit[0], &_jfit[0]);
	_ig = 1;
}

void PikaiaSearch::iterate() {

	// Main Population Loop
	_newtot = 0;
	for (int ip = 1; ip <= _np / 2; ip++) {

		// 1. pick two parents
		select(_np, &_jfit[0], _fdif, _ip1);
		label21: select(_np, &_jfit[0], _fdif, _ip2);
		if (_ip1 == _ip2) {
			goto label21;
		}

		// 2. encode parent phenotypes
		encode(_n, _nd, row_ptr(_ip1, _n, &_oldph[0]), &_gn1[0]);
		encode(_n, _nd, row_ptr(_ip2, _n, &_oldph[0]), &_gn2[0]);

		// 3. breed
		cross(_n, _nd, _pcross, &_gn1[0], &_gn2[0]);
		mutate(_n, _nd, _pmut, &_gn1[0], _imut);
		mutate(_n, _nd, _pmut, &_gn2[0], _imut);

		// 4. decode offspring genotypes
		decode(_n, _nd, &_gn1[0], &_ph[0]);
		decode(_n, _nd, &_gn2[0], &_ph[_n]);

		// 5. insert into population
		if (_irep == 1) {
			genrep(_n, _n, _np, ip, &_ph[0], &_newph[0]);
		} else {
			stdrep(_f._f, _n, _n, _np, _irep, _ielite, &_ph[0], &_oldph[0],
					&_fitns[0], &_ifit[0], &_jfit[0], _new);
			_newtot += _new;
		}

		// End of Main Population Loop
	}

	// if running full generational replacement: swap populations
	if (_irep == 1) {
		newpop(_f._f, _ielite, _n, _n, _np, &_oldph[0], &_newph[0], &_ifit[0],
				&_jfit[0], &_fitns[0], _newtot);
	}

	//  adjust mutation rate?
	if (_imut == 2 || _imut == 3 || _imut == 5 || _imut == 6) {
		adjmut(_n, _n, _np, &_oldph[0], &_fitns[0], &_ifit[0], _pmutmn, _pmutmx,
				_pmut, _imut);
	}

	// End of Main Generation Loop
	_ig++;
}

multivariate_solution PikaiaSearch::solution() {

	// Return best phenotype and its fitness
	std::vector<double> x(_n);
	double *oldph_ptr = row_ptr(_ifit[_np - 1], _n, &_oldph[0]);
	std::copy(oldph_ptr, oldph_ptr + _n, &x[0]);
	return {x, _fev, true};
}

multivariate_solution PikaiaSearch::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);

	// Main Generation Loop
	while (_ig <= _ngen) {
		iterate();
	}

	// Return best phenotype and its fitness
	std::vector<double> x(_n);
	double *oldph_ptr = row_ptr(_ifit[_np - 1], _n, &_oldph[0]);
	std::copy(oldph_ptr, oldph_ptr + _n, &x[0]);
	return {x, _fev, true};
}

double* PikaiaSearch::row_ptr(int row, int n, double *d) {
	return &d[(row - 1) * n];
}

void PikaiaSearch::rqsort(int n, double *a, int *p) {
	const int lgn = 32;
	const int q = 11;
	int s, t, l, m, r, i, j;
	double x;
	std::vector<int> stackl(lgn), stackr(lgn);

	// Initialize the stack
	stackl[0] = 1;
	stackr[0] = n;
	s = 1;

	// Initialize the pointer array
	for (i = 1; i <= n; i++) {
		p[i - 1] = i;
	}

	label2: if (s > 0) {
		l = stackl[s - 1];
		r = stackr[s - 1];
		s--;

		label3: if (r - l < q) {

			// Use straight insertion
			for (i = l + 1; i <= r; i++) {
				t = p[i - 1];
				x = a[t - 1];
				for (j = i - 1; j >= l; j--) {
					if (a[p[j - 1] - 1] <= x) {
						goto label5;
					}
					p[j] = p[j - 1];
				}
				j = l - 1;
				label5: p[j] = t;
			}
		} else {

			// Use quicksort, with pivot as median of a(l), a(m), a(r)
			m = (l + r) / 2;
			t = p[m - 1];
			if (a[t - 1] < a[p[l - 1] - 1]) {
				p[m - 1] = p[l - 1];
				p[l - 1] = t;
				t = p[m - 1];
			}
			if (a[t - 1] > a[p[r - 1] - 1]) {
				p[m - 1] = p[r - 1];
				p[r - 1] = t;
				t = p[m - 1];
				if (a[t - 1] < a[p[l - 1] - 1]) {
					p[m - 1] = p[l - 1];
					p[l - 1] = t;
					t = p[m - 1];
				}
			}

			// Partition
			x = a[t - 1];
			i = l + 1;
			j = r - 1;
			label7: if (i <= j) {
				label8: if (a[p[i - 1] - 1] < x) {
					i++;
					goto label8;
				}
				label9: if (x < a[p[j - 1] - 1]) {
					j--;
					goto label9;
				}
				if (i <= j) {
					t = p[i - 1];
					p[i - 1] = p[j - 1];
					p[j - 1] = t;
					i++;
					j--;
				}
				goto label7;
			}

			// Stack the larger subfile
			s++;
			if (j - l > r - i) {
				stackl[s - 1] = l;
				stackr[s - 1] = j;
				l = i;
			} else {
				stackl[s - 1] = i;
				stackr[s - 1] = r;
				r = j;
			}
			goto label3;
		}
		goto label2;
	}
}

/***********************************************************************
 c                         GENETICS MODULE
 c**********************************************************************
 c
 c     ENCODE:    encodes phenotype into genotype
 c                called by: PIKAIA
 c
 c     DECODE:    decodes genotype into phenotype
 c                called by: PIKAIA
 c
 c     CROSS:     Breeds two offspring from two parents
 c                called by: PIKAIA
 c
 c     MUTATE:    Introduces random mutation in a genotype
 c                called by: PIKAIA
 c
 c     ADJMUT:    Implements variable mutation rate
 c                called by: PIKAIA
 c
 c**********************************************************************/
void PikaiaSearch::encode(int n, int nd, double *ph, int *gn) {
	double z = std::pow(10., nd);
	int ii = 0;
	for (int i = 1; i <= n; i++) {
		int ip = static_cast<int>(ph[i - 1] * z);
		for (int j = nd; j >= 1; j--) {
			gn[ii + j - 1] = ip % 10;
			ip /= 10;
		}
		ii += nd;
	}
}

void PikaiaSearch::decode(int n, int nd, int *gn, double *ph) {
	double z = std::pow(10., -nd);
	int ii = 0;
	for (int i = 1; i <= n; i++) {
		int ip = 0;
		for (int j = 1; j <= nd; j++) {
			ip = 10 * ip + gn[ii + j - 1];
		}
		ph[i - 1] = ip * z;
		ii += nd;
	}
}

void PikaiaSearch::cross(int n, int nd, double pcross, int *gn1, int *gn2) {

	// Use crossover probability to decide whether a crossover occurs
	if (Random::get(0., 1.) < pcross) {

		// Compute first crossover point
		int ispl = static_cast<int>(Random::get(0., 1.) * n * nd) + 1;

		// Now choose between one-point and two-point crossover
		int ispl2;
		if (Random::get(0., 1.) < 0.5) {
			ispl2 = n * nd;
		} else {
			ispl2 = static_cast<int>(Random::get(0., 1.) * n * nd) + 1;
			// Un-comment following line to enforce one-point crossover
			// ispl2=n*nd
			if (ispl2 < ispl) {
				const int itmp = ispl2;
				ispl2 = ispl;
				ispl = itmp;
			}
		}

		// Swap genes from ispl to ispl2
		for (int i = ispl; i <= ispl2; i++) {
			const int t = gn2[i - 1];
			gn2[i - 1] = gn1[i - 1];
			gn1[i - 1] = t;
		}
	}
}

void PikaiaSearch::mutate(int n, int nd, double pmut, int *gn, int imut) {
	int i, j, k, l, ist, inc, loc;

	// Decide which type of mutation is to occur
	if (imut >= 4 && Random::get(0., 1.) <= 0.5) {

		// CREEP MUTATION OPERATOR
		// Subject each locus to random +/- 1 increment at the rate pmut
		for (i = 1; i <= n; i++) {
			for (j = 1; j <= nd; j++) {
				if (Random::get(0., 1.) < pmut) {

					// Construct integer
					loc = (i - 1) * nd + j;
					inc = static_cast<int>(std::round(Random::get(0., 1.))) * 2
							- 1;
					ist = (i - 1) * nd + 1;
					gn[loc - 1] += inc;

					// This is where we carry over the one (up to two digits)
					// first take care of decrement below 0 case
					if (inc < 0 && gn[loc - 1] < 0) {
						if (j == 1) {
							gn[loc - 1] = 0;
						} else {
							for (k = loc; k >= ist + 1; k--) {
								gn[k - 1] = 9;
								gn[k - 2]--;
								if (gn[k - 2] >= 0) {
									goto label4;
								}
							}

							// we popped under 0.00000 lower bound; fix it up
							if (gn[ist - 1] < 0) {
								for (l = ist; l <= loc; l++) {
									gn[l - 1] = 0;
								}
							}
							label4: ;
						}
					}
					if (inc > 0 && gn[loc - 1] > 9) {
						if (j == 1) {
							gn[loc - 1] = 9;
						} else {
							for (k = loc; k >= ist + 1; k--) {
								gn[k - 1] = 0;
								gn[k - 2]++;
								if (gn[k - 2] <= 9) {
									goto label7;
								}
							}

							// we popped over 9.99999 upper bound; fix it up
							if (gn[ist - 1] > 9) {
								for (l = ist; l <= loc; l++) {
									gn[l - 1] = 9;
								}
							}
							label7: ;
						}
					}
				}
			}
		}
	} else {

		// UNIFORM MUTATION OPERATOR
		// Subject each locus to random mutation at the rate pmut
		for (i = 1; i <= n * nd; i++) {
			if (Random::get(0., 1.) < pmut) {
				gn[i - 1] = static_cast<int>(Random::get(0., 1.) * 10.);
			}
		}
	}
}

void PikaiaSearch::adjmut(int ndim, int n, int np, double *oldph, double *fitns,
		int *ifit, double pmutmn, double pmutmx, double &pmut, int imut) {
	const double rdiflo = 0.05, rdifhi = 0.25, delta = 1.5;
	double rdif = 0.;
	if (imut == 2 || imut == 5) {

		// Adjustment based on fitness differential
		rdif = std::fabs(fitns[ifit[np - 1] - 1] - fitns[ifit[np / 2 - 1] - 1])
				/ +(fitns[ifit[np - 1] - 1] + fitns[ifit[np / 2 - 1] - 1]);
	} else if (imut == 3 || imut == 6) {

		// Adjustment based on normalized metric distance
		rdif = 0.;
		double *np_ptr = row_ptr(ifit[np - 1], ndim, oldph);
		double *np2_ptr = row_ptr(ifit[np / 2 - 1], ndim, oldph);
		for (int i = 1; i <= n; i++) {
			rdif += std::pow(np_ptr[i - 1] - np2_ptr[i - 1], 2.);
		}
		rdif = std::sqrt(rdif) / n;
	}

	if (rdif <= rdiflo) {
		pmut = std::min(pmutmx, pmut * delta);
	} else if (rdif >= rdifhi) {
		pmut = std::max(pmutmn, pmut / delta);
	}
}

/***********************************************************************
 c                       REPRODUCTION MODULE
 c**********************************************************************
 c
 c     SELECT:   Parent selection by roulette wheel algorithm
 c               called by: PIKAIA
 c
 c     RNKPOP:   Ranks initial population
 c               called by: PIKAIA, NEWPOP
 c
 c     GENREP:   Inserts offspring into population, for full
 c               generational replacement
 c               called by: PIKAIA
 c
 c     STDREP:   Inserts offspring into population, for steady-state
 c               reproduction
 c               called by: PIKAIA
 c               calls:     FF
 c
 c     NEWPOP:   Replaces old generation with new generation
 c               called by: PIKAIA
 c               calls:     FF, RNKPOP
 c
 c**********************************************************************/
void PikaiaSearch::select(int np, int *jfit, double fdif, int &idad) {
	int np1 = np + 1;
	double dice = Random::get(0., 1.) * np * np1;
	double rtfit = 0.;
	for (int i = 1; i <= np; i++) {
		rtfit = rtfit + np1 + fdif * (np1 - 2 * jfit[i - 1]);
		if (rtfit >= dice) {
			idad = i;
			return;
		}
	}
	// Assert: loop will never exit by falling through
}

void PikaiaSearch::rnkpop(int n, double *arrin, int *indx, int *rank) {

	// Compute the key index
	rqsort(n, arrin, indx);

	// ...and the rank order
	for (int i = 1; i <= n; i++) {
		rank[indx[i - 1] - 1] = n - i + 1;
	}
}

void PikaiaSearch::genrep(int ndim, int n, int np, int ip, double *ph,
		double *newph) {

	// Insert one offspring pair into new population
	int i1 = 2 * ip - 1;
	int i2 = i1 + 1;
	std::copy(ph, ph + n, row_ptr(i1, ndim, newph));
	std::copy(&ph[ndim], &ph[ndim] + n, row_ptr(i2, ndim, newph));
}

void PikaiaSearch::stdrep(const multivariate &ff, int ndim, int n, int np,
		int irep, int ielite, double *ph, double *oldph, double *fitns,
		int *ifit, int *jfit, int &nnew) {
	int i, j, k, i1, if1;
	double fit;

	nnew = 0;
	for (j = 1; j <= 2; j++) {

		// 1. compute offspring fitness (with caller's fitness function)
		fit = ff(&ph[(j - 1) * ndim]);
		_fev++;

		// 2. if fit enough, insert in population
		for (i = np; i >= 1; i--) {
			if (fit > fitns[ifit[i - 1] - 1]) {

				// make sure the phenotype is not already in the population
				if (i < np) {
					double *ifit_ptr = row_ptr(ifit[i], ndim, oldph);
					double *j_ptr = row_ptr(j, ndim, ph);
					for (k = 1; k <= n; k++) {
						if (ifit_ptr[k - 1] != j_ptr[k - 1]) {
							goto label6;
						}
					}
					goto label1;
					label6: ;
				}

				// offspring is fit enough for insertion, and is unique
				// (i) insert phenotype at appropriate place in population
				if (irep == 3) {
					i1 = 1;
				} else if (ielite == 0 || i == np) {
					i1 = static_cast<int>(Random::get(0., 1.) * np) + 1;
				} else {
					i1 = static_cast<int>(Random::get(0., 1.) * (np - 1)) + 1;
				}
				if1 = ifit[i1 - 1];
				fitns[if1 - 1] = fit;
				double *ph_ptr = row_ptr(j, ndim, ph);
				std::copy(ph_ptr, ph_ptr + n, row_ptr(if1, ndim, oldph));

				// (ii) shift and update ranking arrays
				if (i < i1) {

					// shift up
					jfit[if1 - 1] = np - i;
					for (k = i1 - 1; k >= i + 1; k--) {
						jfit[ifit[k - 1] - 1]--;
						ifit[k] = ifit[k - 1];
					}
					ifit[i] = if1;
				} else {

					// shift down
					jfit[if1 - 1] = np - i + 1;
					for (k = i1 + 1; k <= i; k++) {
						jfit[ifit[k - 1] - 1]++;
						ifit[k - 2] = ifit[k - 1];
					}
					ifit[i - 1] = if1;
				}
				nnew++;
				goto label1;
			}
		}
		label1: ;
	}
}

void PikaiaSearch::newpop(const multivariate &ff, int ielite, int ndim, int n,
		int np, double *oldph, double *newph, int *ifit, int *jfit,
		double *fitns, int &nnew) {
	nnew = np;

	// if using elitism, introduce in new population fittest of old
	// population (if greater than fitness of the individual it is to replace)
	if (ielite == 1 && ff(newph) < fitns[ifit[np - 1] - 1]) {
		double *oldph_ptr = row_ptr(ifit[np - 1], ndim, oldph);
		std::copy(oldph_ptr, oldph_ptr + n, newph);
		nnew--;
	}
	_fev++;

	// replace population
	for (int i = 1; i <= np; i++) {
		std::copy(newph, newph + ndim * np, oldph);

		// get fitness using caller's fitness function
		fitns[i - 1] = ff(row_ptr(i, ndim, oldph));
	}
	_fev += np;

	// compute new population fitness rank order
	rnkpop(np, fitns, ifit, jfit);
}
