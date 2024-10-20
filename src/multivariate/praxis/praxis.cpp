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

#include <cmath>
#include <numeric>
#include <cstdint>
#include <iostream>

#include "../../blas.h"
#include "../../random.hpp"

#include "praxis.h"

using Random = effolkronium::random_static;

Praxis::Praxis(double tol, double mstep) {
	_tol = tol;
	_mstep = mstep;
}

void Praxis::init(const multivariate_problem &f, const double *guess) {
	// nothing to do here
}

void Praxis::iterate() {
	// nothing to do here
}

multivariate_solution Praxis::solution(){
	return {{}, 0, false};
}

multivariate_solution Praxis::optimize(const multivariate_problem &f,
		const double *guess) {
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [PRAXIS]: problem constraints will be ignored."
				<< std::endl;
	}
	std::vector<double> x(guess, guess + f._n);
	const int fev = praxis(f._n, &x[0], f._f);
	return {x, fev, false};
}

int Praxis::praxis(int n, double *x, const multivariate &f) {
	bool fk, illc;
	int i, j, jsearch, k, k2, kl, kt, ktm, nits, nl, nf;
	double d2, df, dmin, dn, dni, f1, fx, h, large, ldfac, lds, ldt, m2, m4,
			machep, qa, qb, qc, qd0, qd1, qf1, r, s, scbd, sf, sl, small, t,
			temp, t2, value, vlarge, vsmall;
	double *d, *q0, *q1, *v, *y, *z;

	//  Allocation.
	d = new double[n];
	q0 = new double[n];
	q1 = new double[n];
	v = new double[n * n];
	y = new double[n];
	z = new double[n];

	//  Initialization.
	machep = std::numeric_limits<double>::epsilon();
	small = machep * machep;
	vsmall = small * small;
	large = 1. / small;
	vlarge = 1. / vsmall;
	m2 = std::sqrt(machep);
	m4 = std::sqrt(m2);

	//  Heuristic numbers:
	scbd = 1.;
	illc = false;
	ktm = 1;
	ldfac = (illc) ? 0.1 : 0.01;
	kt = 0;
	nl = 0;
	nf = 1;
	fx = f(x);
	qf1 = fx;
	t = small + std::fabs(_tol);
	t2 = t;
	dmin = small;
	h = _mstep;
	h = std::max(h, 100. * t);
	ldt = h;

	//  The initial set of search directions V is the identity matrix.
	for (j = 0; j < n; j++) {
		std::fill(&v[j * n], &v[j * n] + n, 0.);
		v[j + j * n] = 1.;
	}
	std::fill(d, d + n, 0.);
	qa = 0.;
	qb = 0.;
	qc = 0.;
	qd0 = 0.;
	qd1 = 0.;
	std::copy(x, x + n, q0);
	std::copy(x, x + n, q1);

	//  The main loop starts here.
	for (;;) {
		sf = d[0];
		d[0] = 0.;

		//  Minimize along the first direction V(*,1).
		jsearch = 0;
		nits = 2;
		d2 = d[0];
		s = 0.;
		value = fx;
		fk = false;
		minny(n, jsearch, nits, d2, s, value, fk, f, x, t, h, v, q0, q1, nl, nf,
				dmin, ldt, fx, qa, qb, qc, qd0, qd1);
		d[0] = d2;
		if (s <= 0.) {
			for (i = 0; i < n; i++) {
				v[i] = -v[i];
			}
		}
		if (sf <= 0.9 * d[0] || d[0] <= 0.9 * sf) {
			std::fill(d + 1, d + n, 0.);
		}

		//  The inner loop starts here.
		for (k = 2; k <= n; k++) {
			std::copy(x, x + n, y);
			sf = fx;
			if (0 < kt) {
				illc = true;
			}
			for (;;) {
				kl = k;
				df = 0.;

				//  A random step follows, to avoid resolution valleys.
				if (illc) {
					for (j = 0; j < n; j++) {
						r = Random::get(0., 1.);
						s = (0.1 * ldt + t2 * std::pow(10., kt)) * (r - 0.5);
						z[j] = s;
						daxpym(n, s, v, j * n + 1, x, 1);
					}
					fx = f(x);
					nf++;
				}

				//  Minimize along the "non-conjugate" directions V(*,K),...,V(*,N).
				for (k2 = k; k2 <= n; k2++) {
					sl = fx;
					jsearch = k2 - 1;
					nits = 2;
					d2 = d[k2 - 1];
					s = 0.;
					value = fx;
					fk = false;
					minny(n, jsearch, nits, d2, s, value, fk, f, x, t, h, v, q0,
							q1, nl, nf, dmin, ldt, fx, qa, qb, qc, qd0, qd1);
					d[k2 - 1] = d2;
					if (illc) {
						s = d[k2 - 1] * std::pow(s + z[k2 - 1], 2);
					} else {
						s = sl - fx;
					}
					if (df <= s) {
						df = s;
						kl = k2;
					}
				}

				//  If there was not much improvement on the first try, set
				//  ILLC = true and start the inner loop again.
				if (illc || std::fabs(100. * machep * fx) <= df) {
					break;
				}
				illc = true;
			}

			//  Minimize along the "conjugate" directions V(*,1),...,V(*,K-1).
			for (k2 = 1; k2 < k; k2++) {
				jsearch = k2 - 1;
				nits = 2;
				d2 = d[k2 - 1];
				s = 0.;
				value = fx;
				fk = false;
				minny(n, jsearch, nits, d2, s, value, fk, f, x, t, h, v, q0, q1,
						nl, nf, dmin, ldt, fx, qa, qb, qc, qd0, qd1);
				d[k2 - 1] = d2;
			}
			f1 = fx;
			fx = sf;
			for (i = 0; i < n; i++) {
				temp = x[i];
				x[i] = y[i];
				y[i] = temp - y[i];
			}
			lds = dnrm2(n, y);

			//  Discard direction V(*,kl).
			if (small < lds) {
				for (j = kl - 1; k <= j; j--) {
					std::copy(&v[(j - 1) * n], &v[(j - 1) * n] + n, &v[j * n]);
					d[j] = d[j - 1];
				}
				d[k - 1] = 0.;
				dscal1(n, 1. / lds, y, 1, v, (k - 1) * n + 1);

				//  Minimize along the new "conjugate" direction V(*,k), which is
				//  the normalized vector:  (new x) - (old x).
				jsearch = k - 1;
				nits = 4;
				d2 = d[k - 1];
				value = f1;
				fk = true;
				minny(n, jsearch, nits, d2, lds, value, fk, f, x, t, h, v, q0,
						q1, nl, nf, dmin, ldt, fx, qa, qb, qc, qd0, qd1);
				d[k - 1] = d2;
				if (lds <= 0.) {
					lds = -lds;
					for (i = 0; i < n; i++) {
						v[i + (k - 1) * n] = -v[i + (k - 1) * n];
					}
				}
			}
			ldt = ldfac * ldt;
			ldt = std::max(ldt, lds);
			t2 = dnrm2(n, x);
			t2 = m2 * t2 + t;

			//  See whether the length of the step taken since starting the
			//  inner loop exceeds half the tolerance.
			if (0.5 * t2 < ldt) {
				kt = -1;
			}
			kt++;
			if (ktm < kt) {
				delete[] d;
				delete[] q0;
				delete[] q1;
				delete[] v;
				delete[] y;
				delete[] z;
				return nf;
			}
		}

		//  The inner loop ends here.
		//  Try quadratic extrapolation in case we are in a curved valley.
		quad(n, f, x, t, h, v, q0, q1, nl, nf, dmin, ldt, fx, qf1, qa, qb, qc,
				qd0, qd1);
		for (j = 0; j < n; j++) {
			d[j] = 1. / std::sqrt(d[j]);
		}
		dn = *std::max_element(d, d + n);
		for (j = 0; j < n; j++) {
			dscalm(n, d[j] / dn, v, j * n + 1);
		}

		//  Scale the axes to try to reduce the condition number.
		if (1. < scbd) {
			for (i = 0; i < n; i++) {
				s = 0.;
				for (j = 0; j < n; j++) {
					s = s + v[i + j * n] * v[i + j * n];
				}
				s = std::sqrt(s);
				z[i] = std::max(m4, s);
			}
			s = *std::min_element(z, z + n);
			for (i = 0; i < n; i++) {
				sl = s / z[i];
				z[i] = 1. / sl;
				if (scbd < z[i]) {
					sl = 1. / scbd;
					z[i] = scbd;
				}
				for (j = 0; j < n; j++) {
					v[i + j * n] *= sl;
				}
			}
		}

		//  Calculate a new set of orthogonal directions
		tr_mat(n, v);
		minfit(n, vsmall, v, d);

		//  Unscale the axes.
		if (1. < scbd) {
			for (i = 0; i < n; i++) {
				for (j = 0; j < n; j++) {
					v[i + j * n] *= z[i];
				}
			}
			for (j = 0; j < n; j++) {
				s = dnrm2(n, &v[j * n]);
				d[j] *= s;
				dscalm(n, 1. / s, v, j * n + 1);
			}
		}
		for (i = 0; i < n; i++) {
			dni = dn * d[i];
			if (large < dni) {
				d[i] = vsmall;
			} else if (dni < small) {
				d[i] = vlarge;
			} else {
				d[i] = 1. / dni / dni;
			}
		}

		//  Sort the eigenvalues and eigenvectors.
		svsort(n, d, v);

		//  Determine the smallest eigenvalue.
		dmin = std::max(d[n - 1], small);

		//  The ratio of the smallest to largest eigenvalue determines whether
		//  the system is ill conditioned.
		illc = (dmin < m2 * d[0]);
	}

	delete[] d;
	delete[] q0;
	delete[] q1;
	delete[] v;
	delete[] y;
	delete[] z;
	return nf;
}

double Praxis::flin(int n, int jsearch, double l, const multivariate &f,
		double *x, int &nf, double *v, double *q0, double *q1, double &qd0,
		double &qd1, double &qa, double &qb, double &qc) {
	double *t;
	t = new double[n];

	if (0 <= jsearch) {

		// The search is linear.
		daxpy1(n, l, v, jsearch * n + 1, x, 1, t, 1);
	} else {

		// The search is along a parabolic space curve.
		qa = l * (l - qd1) / (qd0 + qd1) / qd0;
		qb = -(l + qd0) * (l - qd1) / qd1 / qd0;
		qc = (l + qd0) * l / qd1 / (qd0 + qd1);
		for (int i = 0; i < n; i++) {
			t[i] = qa * q0[i] + qb * x[i] + qc * q1[i];
		}
	}

	// The function evaluation counter NF is incremented.
	nf++;

	// Evaluate the function.
	const double value = f(t);
	delete[] t;
	return value;
}

void Praxis::minfit(int n, double tol, double *a, double *q) {
	const int kt_max = 30;
	int i, ii, j, jj, k, kt, l, l2, skip;
	double c, eps, f, g, h, s, temp, x, y, z;
	double *e;

	// Householder's reduction to bidiagonal form.
	f = 0.;
	if (n == 1) {
		q[0] = a[0];
		a[0] = 1.;
		return;
	}

	e = new double[n];
	eps = std::numeric_limits<double>::epsilon();
	g = 0.;
	x = 0.;
	for (i = 1; i <= n; i++) {
		e[i - 1] = g;
		l = i + 1;
		s = 0.;
		for (ii = i; ii <= n; ii++) {
			s = s + a[ii - 1 + (i - 1) * n] * a[ii - 1 + (i - 1) * n];
		}
		g = 0.;
		if (tol <= s) {
			f = a[i - 1 + (i - 1) * n];
			g = std::sqrt(s);
			if (0. <= f) {
				g = -g;
			}
			h = f * g - s;
			a[i - 1 + (i - 1) * n] = f - g;
			for (j = l; j <= n; j++) {
				f = 0.;
				for (ii = i; ii <= n; ii++) {
					f = f + a[ii - 1 + (i - 1) * n] * a[ii - 1 + (j - 1) * n];
				}
				f /= h;
				for (ii = i; ii <= n; ii++) {
					a[ii - 1 + (j - 1) * n] += f * a[ii - 1 + (i - 1) * n];
				}
			}
		}
		q[i - 1] = g;
		s = 0.;
		for (j = l; j <= n; j++) {
			s = s + a[i - 1 + (j - 1) * n] * a[i - 1 + (j - 1) * n];
		}
		g = 0.;
		if (tol <= s) {
			if (i < n) {
				f = a[i - 1 + i * n];
			}
			g = std::sqrt(s);
			if (0. <= f) {
				g = -g;
			}
			h = f * g - s;
			if (i < n) {
				a[i - 1 + i * n] = f - g;
				for (jj = l; jj <= n; jj++) {
					e[jj - 1] = a[i - 1 + (jj - 1) * n] / h;
				}
				for (j = l; j <= n; j++) {
					s = 0.;
					for (jj = l; jj <= n; jj++) {
						s = s
								+ a[j - 1 + (jj - 1) * n]
										* a[i - 1 + (jj - 1) * n];
					}
					for (jj = l; jj <= n; jj++) {
						a[j - 1 + (jj - 1) * n] += s * e[jj - 1];
					}
				}
			}
		}
		y = std::fabs(q[i - 1]) + std::fabs(e[i - 1]);
		x = std::max(x, y);
	}

	// Accumulation of right-hand transformations.
	a[n - 1 + (n - 1) * n] = 1.;
	g = e[n - 1];
	l = n;
	for (i = n - 1; 1 <= i; i--) {
		if (g != 0.) {
			h = a[i - 1 + i * n] * g;
			for (ii = l; ii <= n; ii++) {
				a[ii - 1 + (i - 1) * n] = a[i - 1 + (ii - 1) * n] / h;
			}
			for (j = l; j <= n; j++) {
				s = 0.;
				for (jj = l; jj <= n; jj++) {
					s = s + a[i - 1 + (jj - 1) * n] * a[jj - 1 + (j - 1) * n];
				}
				for (ii = l; ii <= n; ii++) {
					a[ii - 1 + (j - 1) * n] += s * a[ii - 1 + (i - 1) * n];
				}
			}
		}
		for (jj = l; jj <= n; jj++) {
			a[i - 1 + (jj - 1) * n] = 0.;
		}
		for (ii = l; ii <= n; ii++) {
			a[ii - 1 + (i - 1) * n] = 0.;
		}
		a[i - 1 + (i - 1) * n] = 1.;
		g = e[i - 1];
		l = i;
	}

	// Diagonalization of the bidiagonal form.
	eps *= x;
	for (k = n; 1 <= k; k--) {
		kt = 0;
		for (;;) {
			kt++;
			if (kt_max < kt) {
				e[k - 1] = 0.;
				exit(1);
			}
			skip = 0;
			for (l2 = k; 1 <= l2; l2--) {
				l = l2;
				if (std::fabs(e[l - 1]) <= eps) {
					skip = 1;
					break;
				}
				if (1 < l) {
					if (std::fabs(q[l - 2]) <= eps) {
						break;
					}
				}
			}

			//  Cancellation of E(L) if 1 < L.
			if (!skip) {
				c = 0.;
				s = 1.;
				for (i = l; i <= k; i++) {
					f = s * e[i - 1];
					e[i - 1] = c * e[i - 1];
					if (std::fabs(f) <= eps) {
						break;
					}
					g = q[i - 1];

					//  q(i) = h = sqrt(g*g + f*f).
					h = hypot(f, g);
					q[i - 1] = h;
					if (h == 0.) {
						g = 1.;
						h = 1.;
					}
					c = g / h;
					s = -f / h;
				}
			}

			//  Test for convergence for this index K.
			z = q[k - 1];
			if (l == k) {
				if (z < 0.) {
					q[k - 1] = -z;
					for (i = 1; i <= n; i++) {
						a[i - 1 + (k - 1) * n] *= (-1);
					}
				}
				break;
			}

			//  Shift from bottom 2*2 minor.
			x = q[l - 1];
			y = q[k - 2];
			g = e[k - 2];
			h = e[k - 1];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2. * h * y);
			g = hypot(f, 1.);
			if (f < 0.) {
				temp = f - g;
			} else {
				temp = f + g;
			}
			f = ((x - z) * (x + z) + h * (y / temp - h)) / x;

			//  Next QR transformation.
			c = 1.;
			s = 1.;
			for (i = l + 1; i <= k; i++) {
				g = e[i - 1];
				y = q[i - 1];
				h = s * g;
				g *= c;
				z = hypot(f, h);
				e[i - 2] = z;
				if (z == 0.) {
					f = 1.;
					z = 1.;
				}
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = -x * s + g * c;
				h = y * s;
				y *= c;
				for (j = 1; j <= n; j++) {
					x = a[j - 1 + (i - 2) * n];
					z = a[j - 1 + (i - 1) * n];
					a[j - 1 + (i - 2) * n] = x * c + z * s;
					a[j - 1 + (i - 1) * n] = -x * s + z * c;
				}
				z = hypot(f, h);
				q[i - 2] = z;
				if (z == 0.) {
					f = 1.;
					z = 1.;
				}
				c = f / z;
				s = h / z;
				f = c * g + s * y;
				x = -s * g + c * y;
			}
			e[l - 1] = 0.;
			e[k - 1] = f;
			q[k - 1] = x;
		}
	}
	delete[] e;
}

void Praxis::minny(int n, int jsearch, int nits, double &d2, double &x1,
		double &f1, bool fk, const multivariate &f, double *x, double t,
		double h, double *v, double *q0, double *q1, int &nl, int &nf,
		double dmin, double ldt, double &fx, double &qa, double &qb, double &qc,
		double &qd0, double &qd1) {
	int dz, k, ok;
	double d1, f0, f2, fm, m2, m4, machep, s, sf1, small, sx1, t2, temp, x2, xm;

	machep = std::numeric_limits<double>::epsilon();
	small = machep * machep;
	m2 = std::sqrt(machep);
	m4 = std::sqrt(m2);
	sf1 = f1;
	sx1 = x1;
	k = 0;
	xm = 0.;
	fm = fx;
	f0 = fx;
	dz = (d2 < machep);

	//  Find the step size.
	s = dnrm2(n, x);
	if (dz) {
		temp = dmin;
	} else {
		temp = d2;
	}
	t2 = m4 * std::sqrt(std::fabs(fx) / temp + s * ldt) + m2 * ldt;
	s = m4 * s + t;
	if (dz && s < t2) {
		t2 = s;
	}
	t2 = std::max(t2, small);
	t2 = std::min(t2, 0.01 * h);
	if (fk && f1 <= fm) {
		xm = x1;
		fm = f1;
	}
	if ((!fk) || std::fabs(x1) < t2) {
		if (0. <= x1) {
			temp = 1.;
		} else {
			temp = -1.;
		}
		x1 = temp * t2;
		f1 = flin(n, jsearch, x1, f, x, nf, v, q0, q1, qd0, qd1, qa, qb, qc);
	}
	if (f1 <= fm) {
		xm = x1;
		fm = f1;
	}

	//  Evaluate FLIN at another point and estimate the second derivative.
	for (;;) {
		if (dz) {
			if (f1 <= f0) {
				x2 = 2. * x1;
			} else {
				x2 = -x1;
			}
			f2 = flin(n, jsearch, x2, f, x, nf, v, q0, q1, qd0, qd1, qa, qb,
					qc);
			if (f2 <= fm) {
				xm = x2;
				fm = f2;
			}
			d2 = (x2 * (f1 - f0) - x1 * (f2 - f0)) / ((x1 * x2) * (x1 - x2));
		}

		//  Estimate the first derivative at 0.
		d1 = (f1 - f0) / x1 - x1 * d2;
		dz = 1;

		//  Predict the minimum.
		if (d2 <= small) {
			if (0. <= d1) {
				x2 = -h;
			} else {
				x2 = h;
			}
		} else {
			x2 = (-0.5 * d1) / d2;
		}
		if (h < std::fabs(x2)) {
			if (x2 <= 0.) {
				x2 = -h;
			} else {
				x2 = h;
			}
		}

		//  Evaluate F at the predicted minimum.
		ok = 1;
		for (;;) {
			f2 = flin(n, jsearch, x2, f, x, nf, v, q0, q1, qd0, qd1, qa, qb,
					qc);
			if (nits <= k || f2 <= f0) {
				break;
			}
			k++;
			if (f0 < f1 && 0. < x1 * x2) {
				ok = 0;
				break;
			}
			x2 *= 0.5;
		}
		if (ok) {
			break;
		}
	}

	//  Increment the one-dimensional search counter.
	nl++;
	if (fm < f2) {
		x2 = xm;
	} else {
		fm = f2;
	}

	//  Get a new estimate of the second derivative.
	if (small < std::fabs(x2 * (x2 - x1))) {
		d2 = (x2 * (f1 - f0) - x1 * (fm - f0)) / ((x1 * x2) * (x1 - x2));
	} else if (0 < k) {
		d2 = 0.;
	}
	d2 = std::max(d2, small);
	x1 = x2;
	fx = fm;
	if (sf1 < fx) {
		fx = sf1;
		x1 = sx1;
	}

	//  Update X for linear search.
	if (0 <= jsearch) {
		daxpym(n, x1, v, jsearch * n + 1, x, 1);
	}
}

void Praxis::quad(int n, const multivariate &f, double *x, double t, double h,
		double *v, double *q0, double *q1, int &nl, int &nf, double dmin,
		double ldt, double &fx, double &qf1, double &qa, double &qb, double &qc,
		double &qd0, double &qd1) {
	bool fk;
	int i, jsearch, nits;
	double l, s, temp, value;

	temp = fx;
	fx = qf1;
	qf1 = temp;
	for (i = 0; i < n; i++) {
		temp = x[i];
		x[i] = q1[i];
		q1[i] = temp;
	}
	qd1 = 0.;
	for (i = 0; i < n; i++) {
		qd1 = qd1 + (x[i] - q1[i]) * (x[i] - q1[i]);
	}
	qd1 = std::sqrt(qd1);
	if (qd0 <= 0. || qd1 <= 0. || nl < 3 * n * n) {
		fx = qf1;
		qa = 0.;
		qb = 0.;
		qc = 1.;
		s = 0.;
	} else {
		jsearch = -1;
		nits = 2;
		s = 0.;
		l = qd1;
		value = qf1;
		fk = true;
		minny(n, jsearch, nits, s, l, value, fk, f, x, t, h, v, q0, q1, nl, nf,
				dmin, ldt, fx, qa, qb, qc, qd0, qd1);
		qa = l * (l - qd1) / (qd0 + qd1) / qd0;
		qb = -(l + qd0) * (l - qd1) / qd1 / qd0;
		qc = (l + qd0) * l / qd1 / (qd0 + qd1);
	}
	qd0 = qd1;
	for (i = 0; i < n; i++) {
		s = q0[i];
		q0[i] = x[i];
		x[i] = qa * s + qb * x[i] + qc * q1[i];
	}
}

void Praxis::tr_mat(int n, double *a) {
	int i, j;
	double t;
	for (j = 0; j < n; j++) {
		for (i = 0; i < j; i++) {
			t = a[i + j * n];
			a[i + j * n] = a[j + i * n];
			a[j + i * n] = t;
		}
	}
}

void Praxis::svsort(int n, double *d, double *v) {
	int i, j1, j2, j3;
	double t;

	for (j1 = 0; j1 < n - 1; j1++) {

		//  Find J3, the index of the largest entry in D[J1:N-1].
		//  MAXLOC apparently requires its output to be an array.
		j3 = j1;
		for (j2 = j1 + 1; j2 < n; j2++) {
			if (d[j3] < d[j2]) {
				j3 = j2;
			}
		}

		//  If J1 != J3, swap D[J1] and D[J3], and columns J1 and J3 of V.
		if (j1 != j3) {
			t = d[j1];
			d[j1] = d[j3];
			d[j3] = t;
			for (i = 0; i < n; i++) {
				t = v[i + j1 * n];
				v[i + j1 * n] = v[i + j3 * n];
				v[i + j3 * n] = t;
			}
		}
	}
}
