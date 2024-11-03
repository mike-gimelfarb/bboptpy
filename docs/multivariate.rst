Multivariate Optimizers
====================================

The optimization problem for a multivariate function on a box-bounded support can be written as:

.. math::

	\min_{x_1, \dots x_n} f(x_1, \dots x_n), \quad x_i \in [lb_i, ub_i], \, i = 1 \dots n. 
	
bboptpy provides a number of algorithms for optimizing multivariate functions.


Adaptive Coordinate Descent (ACD)
-------------------

This algorithm is described in the following paper:

* Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Adaptive coordinate descent." Proceedings of the 13th annual conference on Genetic and evolutionary computation. 2011.

ACD is a variation of coordinate descent that is more effective for non-separable objectives,
since it maintains a transformation of the coordinate system such that the variables are decorrelated
under the objective function. 

.. function:: ACD(mfev, xtol, ftol, ksucc = 2.0, kunsucc = 0.5)

   Initializes a new ACD optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param xtol: Terminate when the change in solution is less than this value.
   :type xtol: float
   :param ftol: Terminate when the change in objective value is less than this value.
   :type ftol: float
   :param ksucc: Increase factor for step size on successful update.
   :type ksucc: float
   :param kunsucc: Decrease factor for step size on unsuccessful update.
   :type kunsucc: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch

AMaLGaM IDEA
-------------------

This algorithm is described in the following paper:

* Bosman, Peter AN, Jï¿½rn Grahl, and Dirk Thierens. "AMaLGaM IDEAs in noiseless black-box optimization benchmarking." Proceedings of the 11th Annual Conference Companion on Genetic and Evolutionary Computation Conference: Late Breaking Papers. ACM, 2009.

AMaLGaM (Adapted Maximum-Likelihood Gaussian Model Iterated Density-Estimation Evolutionary Algorithm) 
is an estimation-of-distribution (EDA) algorithm that maintains a Gaussian
surrogate function of the optimization landscape that is updated via maximum-likelihood estimation.
It's particularly noted for being parameter-free, meaning it doesn't require manual tuning of parameters to work effectively

.. function:: AMALGAM(mfev, tol, stol, np = 0, iamalgam = True, noparam = True, print = True)

   Initializes a new AMaLGaM optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param stol: Terminate when the standard deviation of population objective values is less than this value.
   :type stol: float
   :param np: Population size (selected automatically when non-positive).
   :type np: int
   :param iamalgam: Whether to use the iAMaLGaM modification with increasing population size and restarts.
   :type iamalgam: bool
   :param noparam: Whether to use the no-parameter version described in the paper.
   :type noparam: bool
   :param print: Whether to print progress on local searches to the console when using the iAMaLGaM version.
   :type print: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Basin Hopping
-------------------

The basin hopping procedure was first outlined in the following paper:

* Wales, David J.; Doye, Jonathan P. K. (1997-07-10). "Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms". The Journal of Physical Chemistry A. 101 (28): 5111–5116.
 
In summary, basin hopping combines two steps in alternation until it achieves a desired global minimum.
First, the algorithm randomly changes the coordinates of the current solution (perturbation).
It then performs a local optimization using some other algorithm. The new coordinates
are accepted if they result in a lower objective value, otherwise they are rejected.
The acceptance/rejection threshold is adjusted during optimization. This algorithm is particularly
useful for optimizing rugged landscapes.

.. function:: BasinHopping(minimizer, stepstrat, print = True, mit = 99, temp = 1.0)

   Initializes a new basin hopping optimizer with the specified parameters.

   :param minimizer: Local optimizer.
   :type minimizer: MultivariateSearch
   :param stepstrat: Strategy for varying step size or acceptance/rejection rate.
   :type stepstrat: BasinHopping_StepStrategy
   :param print: Whether to print progress on local searches to the console.
   :type print: bool
   :param mit: Maximum number of iterations of local search.
   :type mit: int
   :param temp: Initial annealing temperature.
   :type temp: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Controlled Random Search with Local Mutation (CRS)
-------------------

This algorithm (and its other variants) are described in this paper:

* Kaelo, P., and M. M. Ali. "Some variants of the controlled random search algorithm for global optimization." Journal of optimization theory and applications 130.2 (2006): 253-264.

Controlled random search (CRS) works by maintaining a population of solution candidates, where
the worst solutions are replaced with new randomly-generated points. This variant
of CRS improves the local search behavior and efficiency significantly 
by applying a local mutation operator, termed CRS2 in the aforementioned paper.

.. function:: CRS(mfev, np, tol)

   Initializes a new controlled random search optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param np: Population size.
   :type np: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Covariance Matrix Adaptation Evolutionary Strategies (CMA-ES)
-------------------
 
The covariance matrix adaptation evolution strategy (CMA-ES) is a broad family of algorithms
that adapt a mean and covariance matrix to the shape of the search space. It works
by maintaining a population of solution candidates from this distribution, whose best individuals are 
recombined to create new candidate solutions.
 
bboptpy implements many of the best-performing CMA-ES variants.
 
 
Basic CMA-ES
~~~~~~~~
 
The basic CMA-ES was introduced in this paper:
 
* Hansen, Nikolaus, and Andreas Ostermeier. "Completely derandomized self-adaptation in evolution strategies." Evolutionary computation 9.2 (2001): 159-195.
 
The covariance update has ``O(n^3)`` time complexity when updated naively, 
making it scale poorly to higher dimensions. The current implementation applies a 
simple trick of updating the covariance matrix once every ``O(n)`` iterations,
reducing the amortized time complexity to ``O(n^2)``. This allows the algorithm to 
perform well on the order of 100 decision variables.
 
.. function:: CMAES(mfev, tol, np, sigma0 = 2.0, bound = False, eigenrate = 0.25)

   Initializes a new CMA-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param np: Population size.
   :type np: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param bound: Whether to clip sampled candidate solutions to the search space.
   :type bound: bool
   :param eigenrate: Rate at which the covariance matrix is updated.
   :type eigenrate: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Active CMA-ES
~~~~~~~~
 
The active CMA-ES algorithm was introduced in this paper:
 
* Jastrebski, Grahame A., and Dirk V. Arnold. "Improving evolution strategies through active covariance matrix adaptation." Evolutionary Computation, 2006. CEC 2006. IEEE Congress on. IEEE, 2006.
 
In Active CMA-ES, not only is the variance increased in directions 
that have proven successful (as in standard CMA-ES), but also the variance is 
decreased in directions that have been particularly unsuccessful. This greatly
improves the search efficiency.

.. function:: ActiveCMAES(mfev, tol, np, sigma0 = 2.0, bound = False, alphacov = 2.0, eigenrate = 0.25)

   Initializes a new active CMA-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param np: Population size.
   :type np: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param bound: Whether to clip sampled candidate solutions to the search space.
   :type bound: bool
   :param alphacov: the alpha parameter as described in the paper
   :type alphacov: float
   :param eigenrate: Rate at which the covariance matrix is updated.
   :type eigenrate: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Cholesky CMA-ES
~~~~~~~~

The Cholesky variant of CMA-ES was described in this paper:

* Krause, Oswin, Dídac Rodríguez Arbonès, and Christian Igel. "CMA-ES with optimal covariance update and storage complexity." Advances in Neural Information Processing Systems. 2016.

The naive covariance update takes ``O(n^3)`` time. The Cholesky CMA-ES variant
reduces this complexity to ``O(n^2)`` by using the Cholesky decomposition instead
of the eigenvalue decomposition, which often results in a better quality update
than the periodic update of the basic CMA-ES.

.. function:: CholeskyCMAES(mfev, tol, stol, np, sigma0 = 2.0, bound = False)

   Initializes a new Cholesky CMA-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param stol: Terminate when the standard deviation of the population is less than this value.
   :type stol: float
   :param np: Population size.
   :type np: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param bound: Whether to clip sampled candidate solutions to the search space.
   :type bound: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch

 
Limited-Memory CMA-ES
~~~~~~~~

The limited memory variant of CMA-ES was described in this paper:

* Loshchilov, Ilya. "A computationally efficient limited memory CMA-ES for large scale optimization." Proceedings of the 2014 Annual Conference on Genetic and Evolutionary Computation. ACM, 2014. 

The limited-memory CMA-ES variant reduces the complexity of the update
by using a limited number of direction vectors to approximate the covariance matrix.
This allows the algorithm to scale to problems with 1000s or even millions of decision variables.

.. function:: LmCMAES(mfev, tol, np, memory = 0, sigma0 = 2.0, bound = False, rademacher = True, usenew = True)

   Initializes a new limited-memory CMA-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param np: Population size.
   :type np: int
   :param memory: Number of direction vectors in the covariance update (determined automatically when non-positive).
   :type memory: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param bound: Whether to clip sampled candidate solutions to the search space.
   :type bound: bool
   :param rademacher: Whether to sample candidates from Rademacher instead of Gaussian.
   :type rademacher: bool
   :param usenew: Whether to use the new variant of the algorithm from the paper.
   :type usenew: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch

 
Separable CMA-ES
~~~~~~~~

This variant of the CMA-ES was introduced in this paper:

* Ros, Raymond, and Nikolaus Hansen. "A simple modification in CMA-ES achieving linear time and space complexity." International Conference on Parallel Problem Solving from Nature. Springer, Berlin, Heidelberg, 2008.

The separable CMA-ES variant reduces the complexity of the covariance update by
maintaining only a diagonal representation of the covariance matrix. This reduces the cost of the
covariance update to ``O(n)``, making it well-suited for optimizing problems with 1000s or millions
of decision variables. However, this simplification comes at the cost of being able to capture 
complex dependencies between decision variables, making it most suitable for separable objective functions.

.. function:: SepCMAES(mfev, tol, np, sigma0 = 2.0, bound = False, adjustlr = True)

   Initializes a new separable CMA-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param np: Population size.
   :type np: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param bound: Whether to clip sampled candidate solutions to the search space.
   :type bound: bool
   :param adjustlr: Whether to apply the empirical learning rate adjustment in the paper.
   :type adjustlr: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


IPOP-CMA-ES and NIPOP-CMA-ES
~~~~~~~~

The IPOP-CMA-ES and NIPOP-CMA-ES variants of the CMA-ES algorithm were described in these papers:

* Auger, Anne, and Nikolaus Hansen. "A restart CMA evolution strategy with increasing population size." Evolutionary Computation, 2005. The 2005 IEEE Congress on. Vol. 2. IEEE, 2005.
* Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box Optimization Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the BBOB-2012 Noiseless Testbed." Genetic and Evolutionary Computation Conference (GECCO-2012), ACM Press : 269-276. July 2012.

The Increasing Population CMA-ES (IPOP-CMA-ES) implements a multi-restart strategy
for CMA-ES, where the population is increased in each restart. The use of larger population
size makes the search more global, which helps to avoid local minima and to explore the
search space more thoroughly.

.. function:: IPopCMAES(base, mfev, print = false, sigma0 = 2.0, nipop = True, ksigmadec = 1.6, boundlambda = True)

   Initializes a new IPOP-CMA-ES optimizer with the specified parameters.

   :param base: CMA-ES variant to use to optimize in each restart.
   :type base: BaseCMAES
   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param print: Whether to print progress on restarts to the console.
   :type print: bool
   :param sigma0: Initial step size.
   :type sigma0: float
   :param nipop: Whether to use the NIPOP-CMA-ES variant described in the second reference.
   :type nipop: bool
   :param ksigmadec: Factor to increase step size in each restart.
   :type ksigmadec: float   
   :param boundlambda: Whether to apply optimal lambda cycling as described in the paper.
   :type boundlambda: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


BIPOP-CMA-ES and NBIPOP-CMA-ES
~~~~~~~~

The BIPOP-CMA-ES and NBIPOP-CMA-ES variants of the CMA-ES algorithm were described in these papers:

* Hansen, Nikolaus. "Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed." Proceedings of the 11th Annual Conference Companion on Genetic and Evolutionary Computation Conference: Late Breaking Papers. ACM, 2009.
* Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box Optimization Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the BBOB-2012 Noiseless Testbed." Genetic and Evolutionary Computation Conference (GECCO-2012), ACM Press : 269-276. July 2012.

The Bi-Population CMA-ES (BIPOP-CMA-ES) is a variant of the IPOP-CMA-ES that incorporates
two different population sizes to improve the optimization. It uses two restart regimes,
one with a small population size, and one with a larger increasing population
size. It dynamically switches between the two regimes based on the progress of the optimization.

.. function:: BiPopCMAES(base, mfev, print = false, sigma0 = 2.0, maxlargeruns = 9, nbipop = True, ksigmadec = 1.6, kbudget = 2.0)

   Initializes a new BIPOP-CMA-ES optimizer with the specified parameters.

   :param base: CMA-ES variant to use to optimize in each restart.
   :type base: BaseCMAES
   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param print: Whether to print progress on restarts to the console.
   :type print: bool
   :param sigma0: Initial step size.
   :type sigma0: float
   :param maxlargeruns: Maximum number of restarts with large population size.
   :type maxlargeruns: int
   :param nbipop: Whether to use the NBIPOP-CMA-ES variant described in the second reference.
   :type nbipop: bool
   :param ksigmadec: Factor to increase step size in each restart.
   :type ksigmadec: float   
   :param kbudget: How much budget in function evaluations to allocate to the large population size restarts.
   :type kbudget: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Differential Evolution (DE)
-------------------

Differential evolution (DE) is a broad family of algorithms that is inspired by the replication
and mutation of DNA sequences in nature. It maintains a population of candidate solutions that are combined through
mutation and crossover operations to create new candidate solutions, which then replace the existing population through
a process of selection (to mimic the process of survival-of-the-fittest in nature).

bboptpy implements many of the best-performing DE variants from the literature.


JADE
~~~~~~~~

The JADE algorithm was developed in the following series of papers:

* Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: Self-adaptive differential evolution with fast and reliable convergence performance." 2007 IEEE congress on evolutionary computation. IEEE, 2007.
* Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: adaptive differential evolution with optional external archive." IEEE Transactions on evolutionary computation 13.5 (2009): 945-958.
* Li, Jie, et al. "Power mean based crossover rate adaptive differential evolution." International Conference on Artificial Intelligence and Computational Intelligence. Springer, Berlin, Heidelberg, 2011.
* Gong, Wenyin, Zhihua Cai, and Yang Wang. "Repairing the crossover rate in adaptive differential evolution." Applied Soft Computing 15 (2014): 149-168.
 
JADE is an adaptive differential evolution that tunes the crossover and mutation 
rates based on values that performed well in the past. These values are maintained 
as long-run moving averages of the best-performing values from previous iterations.

The current version of JADE implements the optional external archive as described 
in the second paper to improve the population diversity, uses a power mean adaptation for the 
crossover rate as suggested in the third paper, and repairs the crossover
rate as suggested in the fourth paper.

.. function:: JADE(mfev, np, tol, archive = True, repaircr = True, pelite = 0.05, cdamp = 0.1, sigma = 0.07)

   Initializes a new JADE optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param np: Population size.
   :type np: int
   :param tol: Terminate when the std of the candidate solutions is less than this value.
   :type tol: float
   :param archive: Whether to maintain an optional archive.
   :type archive: bool
   :param repaircr: Whether to repair the crossover rate.
   :type repaircr: bool
   :param pelite: Top fraction of candidates to sample from for mutation.
   :type pelite: float
   :param cdamp: Exponential moving average coefficient for updating parameters.
   :type cdamp: float
   :param sigma: Lower bound on std for the power-mean crossover adaptation.
   :type sigma: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


L-SHADE
~~~~~~~~

The L-SHADE algorithm was described in the following paper:

* Tanabe, Ryoji, and Alex S. Fukunaga. "Improving the search performance of SHADE using linear population size reduction." 2014 IEEE congress on evolutionary computation (CEC). IEEE, 2014.

In summary, L-SHADE is very similar to JADE (and in fact builds upon it), but the method of
updating the crossover and mutation rates is different. In L-SHADE, these parameters
are adapted by maintaining a history of H previous parameter values that resulted 
in candidates with good objective values. This history is updated periodically 
and values are drawn from it before each mutation and crossover operation.

The provided version implements the optional external archive as described for JADE,
the linear population size reduction described in the aforementioned 
paper, and the crossover rate repair strategy.

.. function:: SHADE(mfev, npinit, tol, archive = True, repaircr = True, h = 100, npmin = 4)

   Initializes a new L-SHADE optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param npinit: Initial population size.
   :type npinit: int
   :param tol: Terminate when the std of the candidate solutions is less than this value.
   :type tol: float
   :param archive: Whether to maintain an optional archive.
   :type archive: bool
   :param repaircr: Whether to repair the crossover rate.
   :type repaircr: bool
   :param h: Size of the history for updating crossover and mutation parameters.
   :type h: int
   :param npmin: When this value is strictly less than npinit, a linear population size reduction will be used.
   :type npmin: int
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
   

SANSDE
~~~~~~~~

The SANSDE algorithm was described in the following paper:

* Yang, Zhenyu, Ke Tang, and Xin Yao. "Self-adaptive differential evolution with neighborhood search." 2008 IEEE congress on evolutionary computation (IEEE World Congress on Computational Intelligence). IEEE, 2008.

Similar to JADE and L-SHADE, the SANSDE algorithm also implements its own variant
of parameter adaptation for the crossover and mutation. However, it implements two
different possible mutation strategies, and selects between them based on the strategy
that performed well in past iterations.

The provided version of this algorithm also repairs the crossover rate as suggested for JADE. 

.. function:: SANSDE(mfev, np, tol, repaircr = True, crref = 5, pupdate = 50, crupdate = 25)

   Initializes a new SANSDE optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param np: Population size.
   :type np: int
   :param tol: Terminate when the std of the candidate solutions is less than this value.
   :type tol: float
   :param repaircr: Whether to repair the crossover rate.
   :type repaircr: bool
   :param crref: How often to generate a new crossover rate in iterations.
   :type crref: int
   :param pupdate: How often to update the mutation strategy selection parameter.
   :type pupdate: int
   :param crupdate: How often to update the crossover and mutation parameters.
   :type crupdate: int
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Self-Adapting Spherical Search (SSDE)
~~~~~~~~

This algorithm was described in the following papers:

* Kumar, A., Misra, R. K., Singh, D., Mishra, S., & Das, S. (2019). The spherical search algorithm for bound-constrained global optimization problems. Applied Soft Computing, 85, 105734.
* Zhao, J., Zhang, B., Guo, X., Qi, L., & Li, Z. (2022). Self-adapting spherical search algorithm with differential evolution for global optimization. Mathematics, 10(23), 4519.

The spherical search algorithm works by generating trial solutions on the surface
of a hyper-sphere. The algorithm maintains a population of individuals with the 
best objective values, and uses these candidate solutions to generate the new
trial solutions.

The current version implements the full parameter adaptation
described in the first paper, and is identical to the process used in L-SHADE.
This version also has a flag ``usede`` which hybridizes the sphere search
with a differential-evolution update as described in the second paper. This allows
the algorithm to escape local minima better than vanilla spherical search.

.. function:: SSDE(mfev, npinit, tol, patience = 1000, npmin = 4, ptop = 0.11, h = 100, usede = False, repaircr = True)

   Initializes a new SSDE optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param np: Initial population size.
   :type np: int
   :param tol: Terminate when the std of the candidate solutions is less than this value.
   :type tol: float
   :param patience: Number of iterations without improvement before termination.
   :type patience: int
   :param npmin: When this value is strictly less than npinit, a linear population size reduction will be used.
   :type npmin: int
   :param ptop: Fraction of best candidates to sample from during the sphere search update.
   :type ptop: float
   :param h: Size of the history for updating parameters.
   :type h: int
   :param usede: Whether to use the differential evolution hybrid or the original.
   :type usede: bool
   :param repaircr: Whether to repair the crossover rate.
   :type repaircr: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch

   
Differential Search (DSA)
-------------------

This algorithm is introduced in the following paper:

* P. Civicioglu, "Transforming Geocentric Cartesian Coordinates to Geodetic Coordinates by Using Differential Search Algorithm", Computers and Geosciences, 46, 229-247, 2012.

Similar to differential evolution, differential search (DSA) maintains a population of
candidate solutions that are combined to create new solutions using four different strategies. 
These strategies are inspired by the concept of stable motion and migration of 
superorganisms and are somewhat similar to mutation in DE. 

The current implementation is based on the original Matlab code. However, rather than
selecting among the different strategies randomly at uniform, the current implementation
uses the non-stationary multi-armed bandit algorithm Rexp3 to favor the best-performing
strategy over time.
 
.. function:: DSA(mfev, tol, stol, np, adapt = True, nbatch = 100)

   Initializes a new DSA optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the range of objective values among the candidate solutions is less than this value.
   :type tol: float
   :param stol: Terminate when the std of the candidate solutions is less than this value.
   :type stol: float
   :param np: Population size.
   :type np: int
   :param adapt: Whether to adapt the strategy selection using Rexp3, or use random uniform sampling.
   :type adapt: bool
   :param nbatch: The batch size for the strategy selection algorithm.
   :type nbatch: int
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
 

Exponential Natural Evolution Strategy (xNES)
-------------------

This algorithm is described in the following paper:

* Glasmachers, T., Schaul, T., Yi, S., Wierstra, D., & Schmidhuber, J. (2010, July). Exponential natural evolution strategies. In Proceedings of the 12th annual conference on Genetic and evolutionary computation (pp. 393-400).

The Exponential Natural Evolution Strategy (xNES) is closely related to the CMA-ES
algorithm, in which the population is modeled using a multivariate Gaussian distribution.
The key idea of xNES is to adapt the Gaussian sampling distribution using the natural gradient.

.. function:: xNES(mfev, tol, a0 = 1.0, etamu = 1.0)

   Initializes a new xNES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the range of objective values among the candidate solutions is less than this value.
   :type tol: float
   :param a0: Initial diagonal values of the covariance.
   :type a0: float
   :param etamu: The learning rate for the mean.
   :type etamu: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


LIPO with Max Heuristic and Quadratic Trust Region (Max-LIPO-TR)
-------------------
 
This version of the LIPO algorithm is described in the papers:

* Malherbe, Cédric, and Nicolas Vayatis. "Global optimization of Lipschitz functions." International Conference on Machine Learning. PMLR, 2017.
* Zhang, Z. Q., Li, P. J., Li, Q. K., Dong, X., Lu, X. G., & Zhang, Y. F. (2023). Dynamic machine learning global optimization algorithm and its application to aerodynamics. Journal of Propulsion and Power, 39(4), 524-539.

In summary, LIPO is a global search algorithm that optimizes a piecewise-linear surrogate function
under the Lipschitz assumption, where the Lipschitz constant is typically estimated from the history
of previous function evaluations. The Max-LIPO-TR version significantly enhances the original algorithm
by guiding the search via maximization of the surrogate function. To improve local search,
it also builds and optimizes a local quadratic model of the objective function. This version
of the algorithm uses scipy optimizers as the backend.

.. function:: LIPOSearch(mfev, p=0.2, quasi_random=False, kvalues=None, max_sample_iters=100,
                         maxlipo=True, maxlipo_starts=1, maxlipo_method=None, maxlipo_options=None,
                         tr=True, tr_max_pts=0, tr_max_radius=np.inf, tr_method=None, tr_options=None,
                         verbose=False)

   Initializes a new LIPO optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param p: Probability of performing random search at each iteration for exploration.
   :type p: float
   :param quasi_random: Whether random search uses quasi-random sampling.
   :type quasi_random: bool
   :param kvalues: Set of possible Lipschitz constants (uses default setting in the paper if None).
   :type kvalues: List[float]
   :param max_sample_iters: Maximum iterations for sampling an improvement point (ignored if maxlipo is False).
   :type max_sample_iters: int
   :param maxlipo: Whether the surrogate is optimized directly instead of searching for improvement point.
   :type maxlipo: bool
   :param maxlipo_starts: Number of restarts of the surrogate optimizer.
   :type maxlipo_starts: int
   :param maxlipo_method: Scipy optimizer for maxlipo.
   :type maxlipo_method: str
   :param maxlipo_options: Options to pass to the optimizer for maxlipo.
   :type maxlipo_options: Dict
   :param tr: Whether to build and optimize a local quadratic model.
   :type tr: bool
   :param tr_max_pts: Maximum number of points to use for the quadratic model.
   :type tr_max_pts: int
   :param tr_max_radius: Maximum radius around the best point to use for the quadratic model.
   :type tr_max_radius: float
   :param tr_method: Scipy optimizer for the quadratic model.
   :type tr_method: str
   :param tr_options: Options to pass to the optimizer of the quadratic model.
   :type tr_options: Dict
   :param verbose: Whether to print progress to the console.
   :type verbose: bool
   
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch



Novel Self-Adaptive Harmony Search (NSHS)
-------------------

This version of harmony search is described in the following paper:

* Luo, Kaiping. "A Novel Self-Adaptive Harmony Search Algorithm." Journal of Applied Mathematics 2013.1 (2013): 653749.

Harmony search is loosely inspired by the musical improvisation process of musicians, 
and mimics the way musicians search for a perfect harmony by adjusting the pitch of their instruments.
It is population-based, like some of the previously-described algorithms, where each candidate solution is adjusted
based on a harmony memory. The Novel Self-Adaptive Harmony Search (NSHS) variant adapts the pitch and rate adjustment parameters
automatically based on historical information and is much more efficient than standard harmony search and many
of its variants.

.. function:: NSHS(mfev, hms, fstdmin = 0.0001)

   Initializes a new NSHS optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param hms: Harmony memory size.
   :type hms: int
   :param fstdmin: Lower bound on standard deviation as described in the paper (where it is hard coded as 0.0001).
   :type fstdmin: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
 
 
Hessian Estimation Evolutionary Strategy (HE-ES)
-------------------

This algorithm was introduced in the following paper:

* Glasmachers, Tobias, and Oswin Krause. "The Hessian Estimation Evolution Strategy." International Conference on Parallel Problem Solving from Nature. Springer, Cham, 2020.

Similar to CMA-ES and variants, the Hessian Estimation Evolution Strategy (HE-ES) maintains a Gaussian
sampling distribution by updating its mean and covariance parameters. Crucially in HE-ES,
the covariance matrix is updated using curvature information from the objective function's imputed Hessian.
In practice, HE-ES often performs superior to CMA-ES in terms of robustness and 
number of function evaluations, in many cases even when the objective function
is not twice-differentiable.

.. function:: HEES(mfev, tol, mres = 1, print = False, np = 0, sigma0 = 2.0)

   Initializes a new HE-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the range of objective values among the candidate solutions is less than this value.
   :type tol: float
   :param mres: Number of runs with increasing population size as suggested in the paper.
   :type mres: int
   :param print: Whether to print progress on local searches to the console when using HE-ES with multiple restarts.
   :type print: bool
   :param np: Initial population size.
   :type np: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param fstdmin: Lower bound on standard deviation as described in the paper (where it is hard coded as 0.0001).
   :type fstdmin: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
 

Self-Adaptive Multi-Population JAYA
-------------------

This version of the JAYA search algorithm is described in detail in the following paper:

* Yu, Jiang-Tao, et al. "Jaya algorithm with self-adaptive multi-population and Lévy flights for solving economic load dispatch problems." IEEE Access 7 (2019): 21372-21384.

The JAYA algorithm is a population-based metaheuristic search algorithm for finding
the global minimum of a multivariate function. It is similar to differential evolution,
but the procedure for generating new candidate solutions includes not only an attraction 
term for moving towards the best candidate in the population, but also a repulsion
term for moving away from the worst candidate. The version implemented currently
features a multi-population scheme for improving the search behavior of JAYA,
automatic parameter adaptation, and Levy flights and other mutation operators 
for global exploration of the search space.

.. function:: JAYA(mfev, tol, np, npmin, adapt = True, k0 = 2, mutation = JAYA_Mutation.logistic, scale = 0.01, beta = 1.5, kcheb = 2, temper = 10.0)

   Initializes a new adaptive JAYA optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param np: Total population size.
   :type np: int
   :param npmin: Minimum subpopulation size.
   :type npmin: int
   :param adapt: Whether to adapt the number of subpopulations.
   :type adapt: bool
   :param k0: Initial number of subpopulations.
   :type k0: int
   :param mutation: The type of mutation operator to use.
   :type mutation: JAYA_Mutation
   :param scale: Step size parameter used when the mutation operator is ``levy``.
   :type scale: float
   :param beta: Distributional parameter used when the mutation operator is ``levy``.
   :type beta: float
   :param kcheb: Currently unused.
   :type kcheb: int
   :param temper: Temperature parameter that controls parameter adaptation speed.
   :type temper: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
 
 
Adaptive Nelder-Mead (Simplex) Algorithm
-------------------

The versions of Nelder-Mead implemented are described in the following papers:

* Gao, Fuchang & Han, Lixing. (2012). Implementing the Nelder-Mead simplex algorithm with adaptive parameters. Computational Optimization and Applications. 51. 259-277. 10.1007/s10589-010-9329-3.
* Mehta, V. K. "Improved Nelder–Mead algorithm in high dimensions with adaptive parameters based on Chebyshev spacing points." Engineering Optimization 52.10 (2020): 1814-1828.

The Nelder-Mead algorithm (or the downhill simplex algorithm) is suitable for 
unconstrained optimization problems where the objective function may have noise or 
discontinuities. It maintains a simplex of points that are updated through a series
of transformations, namely: reflection, expansion, contraction, and shrink.

The version of Nelder-Mead provided is based loosely on the original FORTRAN implementations,
but implements a variety of improvements to make the algorithm work better in high dimensions 
and converge faster. These include adaptation of the transformation hyper-parameters,
better simplex initialization, periodic restarts, and better termination criteria.

.. function:: NelderMead(mfev, tol, rad0, minit = NelderMead_SimplexInit.spendley, pinit = NelderMead_ParamInit.mehta2019_refined, checkev = 10, eps = 1e-3)

   Initializes a new adaptive Nelder-Mead optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in solution is less than this value.
   :type tol: float
   :param rad0: Initial scale of the simplex.
   :type rad0: float
   :param minit: Initialization method for the simplex.
   :type minit: NelderMead_SimplexInit
   :param pinit: Transformation hyper-parameter adaptation method.
   :type pinit: NelderMead_ParamInit
   :param checkev: How often to check termination condition.
   :type checkev: int
   :param eps: Small constant used in the factorial test to check termination.
   :type eps: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Particle Swarm Optimization (PSO)
-------------------

Particle swarm optimization is a broad family of global optimization algorithms
that imitates the way groups of animals (e.g. birds, fish) behave. Similar to 
differential evolution and similar metaheuristics, it maintains a population of
candidate solutions (called particles). However, unlike DE, it updates these 
particles using an estimate of the particles' velocity in addition to their 
position in the search space. The position of each particle is updated based
on the particle's personal best position, as well as the global best position
of the entire population. 

bboptpy implements a number of different variants of PSO, typically with parameter
adaptation and other tricks to make them converge more effectively in 
complex, high-dimensional problems.


Adaptive PSO (APSO)
~~~~~~~~

This version of PSO is described in the following paper:

* Zhan, Zhi-Hui, Jun Zhang, Yun Li, and Henry Shu-Hung Chung. "Adaptive particle swarm optimization." IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 39, no. 6 (2009): 1362-1381.

The Adaptive PSO (APSO) algorithm estimates the key hyper-parameters used in the
PSO update equation using a sophisticated approach that adapts its behavior
depending on the phase of the optimization, termed exploration, exploitation, convergence
and jumping out. This helps the algorithm maintain diversity of the population
pool and to avoid local minima.

.. function:: APSO(mfev, tol, np, correct = True)

   Initializes a new APSO optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the standard deviation in population candidates is less than this value.
   :type tol: float
   :param np: Population size.
   :type np: int
   :param correct: Whether to correct particles that go out of the search bounds.
   :type correct: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Competitive PSO (CSO)
~~~~~~~~

This variant was described in the following papers:

* Cheng, Ran & Jin, Yaochu. (2014). A Competitive Swarm Optimizer for Large Scale Optimization. IEEE transactions on cybernetics. 45. 10.1109/TCYB.2014.2322602.
* Mohapatra, Prabhujit, Kedar Nath Das, and Santanu Roy. "A modified competitive swarm optimizer for large scale optimization problems." Applied Soft Computing 59 (2017): 340-362.
* Mohapatra, Prabhujit, Kedar Nath Das, and Santanu Roy. "Inherited competitive swarm optimizer for large-scale optimization problems." Harmony Search and Nature Inspired Optimization Algorithms. Springer, Singapore, 2019. 85-95.

The Competitive PSO (CSO) variant was designed to mimic competition among individuals in a population.
It modifies the PSO update where the winning particles update their position and velocity based on
their objective values, while the losing particles adapt to improve by learning from the winning
particles. This variant is better-suited for large-scale optimization problems than standard PSO, 
by subdividing the population into subpopulations during optimization.

.. function:: CSO(mfev, stol, np, pcompete = 3, ring = False, correct = True, vmax = 0.2)

   Initializes a new CSO optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param stol: Terminate when the standard deviation in population candidates is less than this value.
   :type stol: float
   :param np: Total population size (will be increased if ``pcompete`` does not divide it).
   :type np: int
   :param pcompete: Number of competing subpopulations.
   :type pcompete: int
   :param ring: Whether each particle updates its parameters based on its neighbors (ring topology), or the entire population.
   :type ring: bool
   :param correct: Whether to correct particles that go out of the search bounds.
   :type correct: bool
   :param vmax: Maximum bound on each component of the velocity vector.
   :type vmax: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Cooperative Coevolutionary PSO (CCPSO)
~~~~~~~~

This algorithm was described in the following papers:

* Van den Bergh, Frans, and Andries Petrus Engelbrecht. "A cooperative approach to particle swarm optimization." IEEE transactions on evolutionary computation 8.3 (2004): 225-239.
* Li, Xiaodong, and Xin Yao. "Tackling high dimensional nonseparable optimization problems by cooperatively coevolving particle swarms." 2009 IEEE congress on evolutionary computation. IEEE, 2009.
* Li, Xiaodong, and Xin Yao. "Cooperatively coevolving particle swarms for large scale optimization." IEEE Transactions on Evolutionary Computation 16.2 (2012): 210-224.

The Cooperative Coevolutionary PSO (CCPSO) variant was designed to handle large-scale
optimization problems with large numbers of decision variables. In order to achieve
this, CCPSO first divides the population into subpopulations that co-evolve in parallel.
The CCPSO algorithm also dynamically adjust the influence of each subpopulation based on its
importance, and periodically regroups the population into subpopulations to avoid
premature convergence.

.. function:: CCPSO(mfev, sigmatol, np, pps, npps, correct = True, pcauchy = -1.0, local = None, localfreq = 10)

   Initializes a new CCPSO optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param sigmatol: Terminate when the standard deviation in population candidates is less than this value.
   :type sigmatol: float
   :param np: Total population size.
   :type np: int
   :param pps: Collection of subpopulation sizes (each size must divide ``np``).
   :type pps: List of int
   :param npps: Number of elements in ``pps``.
   :type npps: int
   :param correct: Whether to correct particles that go out of the search bounds.
   :type correct: bool
   :param pcauchy: Probability of sampling candidate from a Cauchy distribution to encourage exploration (adapted if non-positive).
   :type pcauchy: float
   :param local: Local optimizer to perform optional local search after each iteration.
   :type local: MultivariateSearch
   :param localfreq: How often to perform a local search when ``local`` is specified.
   :type localfreq: int
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Self-Learning PSO (SLPSO)
~~~~~~~~

This variant of PSO was described in the following paper:

* Li, Changhe, Shengxiang Yang, and Trung Thanh Nguyen. "A self-learning particle swarm optimizer for global optimization problems." IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 42.3 (2011): 627-646.

The key idea behind Self-Learning PSO (SLPSO) is to equip each particle with multiple learning strategies, 
allowing it to adapt its behavior based on the specific situation it encounters in the search space.
It maintains four key update strategies to achieve this: exploitation, jumping out, exploration and 
convergence, and adapts the selection of these strategies over time based on their success in
improving the objective value. This variant tends to perform well across a broad range of problems.

.. function:: SLPSO(mfev, stol, np, omegamin = 0.4, omegamax = 0.9, eta = 1.496, gamma = 0.01, vmax = 0.2, Ufmax = 10.0)

   Initializes a new SLPSO optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param stol: Terminate when the standard deviation in population candidates is less than this value.
   :type stol: float
   :param np: Population size.
   :type np: int
   :param omegamin: Lower bound on the inertia weight.
   :type omegamin: float
   :param omegamax: Upper bound on the inertia weight.
   :type omegamax: float
   :param eta: Learning rate for velocity update.
   :type eta: float
   :param gamma: Weight decay factor for strategy probability update.
   :type gamma: float
   :param vmax: Maximum bound on each component of the velocity vector.
   :type vmax: float
   :param Ufmax: Maximum bound on the U_f parameter in the paper.
   :type Ufmax: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Powell's Methods
-------------------

Several of M.J.D. Powell's optimization methods are implemented. The references
for this body of work can be found in:

* Powell, Michael JD. "The BOBYQA algorithm for bound constrained optimization without derivatives." Cambridge NA Report NA2009/06, University of Cambridge, Cambridge 26 (2009): 26-46.
* Powell, Michael JD. "The NEWUOA software for unconstrained optimization without derivatives." Large-scale nonlinear optimization (2006): 255-297.


BOBYQA
~~~~~~~~

BOBYQA stands for Bound Optimization BY Quadratic Approximation. It maintains a
quadratic model of the objective function, which is used to derive a trust region
within which to perform optimization. The trust region radius and interpolation
points are chosen adaptively to improve performance. This algorithm is incredibly
robust across a variety of problems, particularly when derivative information
is not available. It is most suitable for functions with at most a few hundred
decision variables, and where bound constraints are provided on the search space.

.. function:: BOBYQA(mfev, np, rho, tol)

   Initializes a new BOBYQA optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param np: Number of interpolation points.
   :type np: int
   :param rho: Initial radius of the trust region.
   :type rho: float
   :param tol: Tolerance to determinate whether the algorithm should terminate.
   :type tol: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


NEWUOA
~~~~~~~~

NEWUOA stands for New Unconstrained Optimization with Quadratic Approximation.
Similar to BOBYQA it maintains a quadratic surrogate model of the objective function, 
and an adaptive trust region is used to select points. Unlike BOBYQA, NEWUOA
restricts the search to the bound constraints provided in the problem.

.. function:: NEWUOA(mfev, np, rho, tol)

   Initializes a new NEWUOA optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param np: Number of interpolation points.
   :type np: int
   :param rho: Initial radius of the trust region.
   :type rho: float
   :param tol: Tolerance to determinate whether the algorithm should terminate.
   :type tol: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


PRAXIS
-------------------

This algorithm is described in the following paper:

* Gegenfurtner, Karl R. "PRAXIS: Brent’s algorithm for function minimization." Behavior Research Methods, Instruments, & Computers 24 (1992): 560-564.

The PRAXIS algorithm was developed by Richard Brent and is a refinement of Powell's
method of conjugate directions. In each iteration, it generates a set of conjugate
directions along which to search for a minimum, and performs a line search along each
direction. It updates the search directions iteratively until a minimum has been found.
It is particularly useful when derivative information is not available, and for
optimizing functions with at most a few hundred decision variables.

.. function:: PRAXIS(tol, mstep)

   Initializes a new NEWUOA optimizer with the specified parameters.

   :param tol: Tolerance to determinate whether the algorithm should terminate.
   :type tol: float
   :param mstep: Maximum step size of line search.
   :type mstep: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Rosenbrock's Method
-------------------

This algorithm and its improvement was described in the following papers:

* Palmer, J. R. "An improved procedure for orthogonalising the search vectors in Rosenbrock's and Swann's direct search optimisation methods." The Computer Journal 12.1 (1969): 69-71.

The Rosenbrock's method uses orthogonal search directions to explore the search space.
Like PRAXIS, it searches along the search directions for a minimum, but originally
used a pattern search instead of a line search. The provided variant implements
this method but replaces the pattern search with a fast Lagrange quadratic interpolation line search
procedure outlined in the paper above. It performs roughly on par with PRAXIS on many problems.

.. function:: Rosenbrock(mfev, tol, step0, decf = 0.1)

   Initializes a new Rosenbrock optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Tolerance to determinate whether the algorithm should terminate.
   :type tol: float
   :param step0: Initial step size of line search.
   :type step0: float
   :param decf: How fast to decay the step size in line search.
   :type decf: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
