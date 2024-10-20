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

 [1] Zervoudakis, Konstantinos, and Stelios Tsafarakis. "A mayfly optimization algorithm."
 Computers & Industrial Engineering 145 (2020): 106559.
 */

#include <cmath>
#include <numeric>
#include <iostream>

#include "../../random.hpp"

#include "mayfly.h"

using Random = effolkronium::random_static;

MayflySearch::MayflySearch(int np, int mfev, double a1, double a2, double a3,
		double beta, double dance, double ddamp, double fl, double fldamp,
		double gmin, double gmax, double vdamp, double sigma, double pmutdim,
		double pmutnp, double l, bool pgb) {
	_np = np;
	_mfev = mfev;
	_a1 = a1;
	_a2 = a2;
	_a3 = a3;
	_beta = beta;
	_dance0 = dance;
	_ddamp = ddamp;
	_fl0 = fl;
	_fldamp = fldamp;
	_gmin = gmin;
	_gmax = gmax;
	_vdamp = vdamp;
	_sigma = sigma;
	_mu = pmutdim;
	_pmut = pmutnp;
	_l = l;
	_pgb = pgb;
}

void MayflySearch::init(const multivariate_problem &f, const double *guess) {

	// initialize problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [Mayfly]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// initialize swarms
	_fbest = std::numeric_limits<double>::infinity();
	_best = std::vector<double>(_n);
	_males.clear();
	_females.clear();
	_offspring.clear();
	for (int j = 0; j < _np; j++) {

		// initialize male
		std::vector<double> mx(_n);
		std::vector<double> mv(_n);
		std::vector<double> mbx(_n);
		for (int i = 0; i < _n; i++) {
			mx[i] = Random::get(_lower[i], _upper[i]);
			mv[i] = 0.;
			mbx[i] = mx[i];
		}
		const double mf = _f._f(&mx[0]);
		const auto &pmFly = std::make_shared<mayfly>(mayfly { mv, mx, mbx, mf,
				mf });
		_males.push_back(std::move(pmFly));

		// update global best
		if (mf < _fbest) {
			_fbest = mf;
			std::copy(mx.begin(), mx.end(), _best.begin());
		}

		// initialize female
		std::vector<double> fx(_n);
		std::vector<double> fv(_n);
		std::vector<double> fxb(_n);
		for (int i = 0; i < _n; i++) {
			fx[i] = Random::get(_lower[i], _upper[i]);
			fv[i] = 0.;
		}
		const double ff = _f._f(&fx[0]);
		const auto &pfFly = std::make_shared<mayfly>(mayfly { fv, fx, fxb, ff,
				ff });
		_females.push_back(std::move(pfFly));

		// update global best (PGB-IMA)
		if (_pgb) {
			if (ff < _fbest) {
				_fbest = ff;
				std::copy(fx.begin(), fx.end(), _best.begin());
			}
		}

		// initialize offspring
		std::vector<double> ox(_n);
		std::vector<double> ov(_n);
		std::vector<double> obx(_n);
		const auto &poFly = std::make_shared<mayfly>(mayfly { ox, ov, obx, 0.,
				0. });
		_offspring.push_back(std::move(poFly));
	}
	_fev = _np + _np;

	// extra swarm for mutation
	_nmut = static_cast<int>(_pmut * _np);
	if (_nmut % 2 != 0) {
		_nmut++;
		_nmut = std::min(_nmut, _np);
	}
	_mutated.clear();
	for (int j = 0; j < _nmut; j++) {
		std::vector<double> mx(_n);
		std::vector<double> mv(_n);
		std::vector<double> mbx(_n);
		const auto &pmFly = std::make_shared<mayfly>(mayfly { mx, mv, mbx, 0.,
				0. });
		_mutated.push_back(std::move(pmFly));
	}

	// initialize other memory
	_indices = std::vector<int>(_n);
	for (int i = 0; i < _n; i++) {
		_indices[i] = i;
	}
	_temp = std::vector<std::shared_ptr<mayfly>>(_np + _np / 2 + _nmut / 2);

	// initialize parameters
	_it = 0;
	_itmax = static_cast<int>(std::ceil((_mfev - _fev) / (3. * _np + _nmut)));
	_g = _gmax;
	_dance = _dance0;
	_fl = _fl0;
}

void MayflySearch::iterate() {

	// update female swarm (_np)
	for (int j = 0; j < _np; j++) {
		auto &female = _females[j];
		updateFemaleVelocity(*female, *_males[j]);
		updatePosition(*female);

		// update global best (PGB-IMA)
		if (_pgb) {
			if (female->_f < _fbest) {
				_fbest = female->_f;
				std::copy(female->_x.begin(), female->_x.end(), _best.begin());
			}
		}
	}

	// update male swarm (_np)
	for (int j = 0; j < _np; j++) {
		auto &male = _males[j];
		updateMaleVelocity(*male);
		updatePosition(*male);

		// update personal best
		if (male->_f < male->_bf) {
			male->_bf = male->_f;
			std::copy(male->_x.begin(), male->_x.end(), male->_bx.begin());
		}

		// update global best
		if (male->_bf < _fbest) {
			_fbest = male->_bf;
			std::copy(male->_bx.begin(), male->_bx.end(), _best.begin());
		}
	}

	// sort the males and female swarms by fitness
	std::sort(_males.begin(), _males.end(), mayfly::compare_fitness);
	std::sort(_females.begin(), _females.end(), mayfly::compare_fitness);

	// mating ritual (_np)
	for (int j = 0; j < _np / 2; j++) {

		// apply crossover operation
		auto &off1 = _offspring[2 * j];
		auto &off2 = _offspring[2 * j + 1];
		crossover(*_males[j], *_females[j], *off1, *off2);

		// update the global best
		for (int k = 2 * j; k <= 2 * j + 1; k++) {
			auto &offk = _offspring[k];
			if (offk->_f < _fbest) {
				_fbest = offk->_f;
				std::copy(offk->_x.begin(), offk->_x.end(), _best.begin());
			}
		}
	}

	// mutation (_nmut)
	Random::shuffle(_offspring.begin(), _offspring.end());
	if (_nmut > 0) {
		for (int j = 0; j < _nmut; j++) {

			// apply mutation operation
			auto &off = _mutated[j];
			mutation(*_offspring[j], *off);

			// update the global best
			if (off->_f < _fbest) {
				_fbest = off->_f;
				std::copy(off->_x.begin(), off->_x.end(), _best.begin());
			}
		}
		Random::shuffle(_mutated.begin(), _mutated.end());
	}

	// rank the male population
	std::copy(_males.begin(), _males.end(), _temp.begin());
	std::copy(_offspring.begin(), _offspring.begin() + _np / 2,
			_temp.begin() + _np);
	if (_nmut > 0) {
		std::copy(_mutated.begin(), _mutated.begin() + _nmut / 2,
				_temp.begin() + _np + _np / 2);
	}
	std::sort(_temp.begin(), _temp.end(), mayfly::compare_fitness);
	for (int j = 0; j < _np; j++) {
		auto &male = _males[j];
		auto &temp = _temp[j];
		std::copy(temp->_x.begin(), temp->_x.end(), male->_x.begin());
		std::copy(temp->_v.begin(), temp->_v.end(), male->_v.begin());
		std::copy(temp->_bx.begin(), temp->_bx.end(), male->_bx.begin());
		male->_f = temp->_f;
		male->_bf = temp->_bf;
	}

	// rank the female population
	std::copy(_females.begin(), _females.end(), _temp.begin());
	std::copy(_offspring.begin() + _np / 2, _offspring.end(),
			_temp.begin() + _np);
	if (_nmut > 0) {
		std::copy(_mutated.begin() + _nmut / 2, _mutated.end(),
				_temp.begin() + _np + _np / 2);
	}
	std::sort(_temp.begin(), _temp.end(), mayfly::compare_fitness);
	for (int j = 0; j < _np; j++) {
		auto &female = _females[j];
		auto &temp = _temp[j];
		std::copy(temp->_x.begin(), temp->_x.end(), female->_x.begin());
		std::copy(temp->_v.begin(), temp->_v.end(), female->_v.begin());
		female->_f = temp->_f;
	}

	// update parameters
	_it++;
	_g = _gmax - ((_gmax - _gmin) / _itmax) * _it;
	_dance *= _ddamp;
	_fl *= _fldamp;
}

multivariate_solution MayflySearch::solution(){
	return {_best, _fev, false};
}

multivariate_solution MayflySearch::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	while (_fev < _mfev) {
		iterate();
	}
	return {_best, _fev, false};
}

/* =============================================================
 *
 * 				FIREFLY MOVEMENT SUBROUTINES
 *
 * =============================================================
 */
void MayflySearch::updateMaleVelocity(mayfly &fly) {

	// compute r_p and r_g
	double rp = 0.;
	double rg = 0.;
	for (int i = 0; i < _n; i++) {
		rp += (fly._bx[i] - fly._x[i]) * (fly._bx[i] - fly._x[i]);
		rg += (_best[i] - fly._x[i]) * (_best[i] - fly._x[i]);
	}

	// calculate velocity
	if (fly._f > _fbest) {
		for (int i = 0; i < _n; i++) {
			fly._v[i] = _g * fly._v[i]
					+ _a1 * std::exp(-_beta * rp) * (fly._bx[i] - fly._x[i])
					+ _a2 * std::exp(-_beta * rg) * (_best[i] - fly._x[i]);
			const double vmax = _vdamp * (_upper[i] - _lower[i]);
			fly._v[i] = std::max(-vmax, std::min(fly._v[i], vmax));
		}
	} else {
		for (int i = 0; i < _n; i++) {
			fly._v[i] = _g * fly._v[i] + _dance * Random::get(-1., 1.);
			const double vmax = _vdamp * (_upper[i] - _lower[i]);
			fly._v[i] = std::max(-vmax, std::min(fly._v[i], vmax));
		}
	}
}

void MayflySearch::updateFemaleVelocity(mayfly &female, mayfly &male) {

	// compute r_mf
	double r = 0.;
	for (int i = 0; i < _n; i++) {
		r += (female._x[i] - male._x[i]) * (female._x[i] - male._x[i]);
	}

	// calculate velocity
	if (male._f < female._f) {
		for (int i = 0; i < _n; i++) {
			female._v[i] = _g * female._v[i]
					+ _a3 * std::exp(-_beta * r) * (male._x[i] - female._x[i]);
			const double vmax = _vdamp * (_upper[i] - _lower[i]);
			female._v[i] = std::max(-vmax, std::min(female._v[i], vmax));
		}
	} else {
		for (int i = 0; i < _n; i++) {
			female._v[i] = _g * female._v[i] + _fl * Random::get(-1., 1.);
			const double vmax = _vdamp * (_upper[i] - _lower[i]);
			female._v[i] = std::max(-vmax, std::min(female._v[i], vmax));
		}
	}
}

void MayflySearch::updatePosition(mayfly &fly) {

	// update position from velocity
	for (int i = 0; i < _n; i++) {
		fly._x[i] += fly._v[i];
		fly._x[i] = std::max(_lower[i], std::min(fly._x[i], _upper[i]));
	}

	// update fitness
	fly._f = _f._f(&(fly._x)[0]);
	_fev++;
}

/* =============================================================
 *
 * 				OFFSPRING GENERATION SUBROUTINES
 *
 * =============================================================
 */
void MayflySearch::crossover(mayfly &par1, mayfly &par2, mayfly &off1,
		mayfly &off2) {

	// crossover operation
	for (int i = 0; i < _n; i++) {
		off1._x[i] = _l * par1._x[i] + (1. - _l) * par2._x[i];
		off1._x[i] = std::max(_lower[i], std::min(off1._x[i], _upper[i]));
		off2._x[i] = _l * par2._x[i] + (1. - _l) * par1._x[i];
		off2._x[i] = std::max(_lower[i], std::min(off2._x[i], _upper[i]));
	}
	std::copy(off1._x.begin(), off1._x.end(), off1._bx.begin());
	std::copy(off2._x.begin(), off2._x.end(), off2._bx.begin());

	// update fitness
	off1._f = _f._f(&(off1._x)[0]);
	off1._bf = off1._f;
	off2._f = _f._f(&(off2._x)[0]);
	off2._bf = off2._f;
	_fev += 2;

	// initialize velocity
	std::fill(off1._v.begin(), off1._v.end(), 0.);
	std::fill(off2._v.begin(), off2._v.end(), 0.);
}

bool MayflySearch::mutation(mayfly &par, mayfly &off) {

	// how many genes to mutate
	const int nmut = static_cast<int>(std::ceil(_mu * _n));

	// apply mutation operation
	Random::shuffle(_indices.begin(), _indices.end());
	for (int i = 0; i < _n; i++) {
		const int k = _indices[i];
		if (i < nmut) {
			const double sigma = _sigma * (_upper[k] - _lower[k]);
			off._x[k] = par._x[k] + sigma * Random::get(_Z);
			off._x[k] = std::max(_lower[k], std::min(off._x[k], _upper[k]));
		} else {
			off._x[k] = par._x[k];
		}
	}
	std::copy(off._x.begin(), off._x.end(), off._bx.begin());

	// update fitness
	const double oldf = par._f;
	off._f = _f._f(&off._x[0]);
	off._bf = off._f;
	_fev++;

	// initialize velocity
	std::fill(off._v.begin(), off._v.end(), 0.);
	return off._f < oldf;
}
