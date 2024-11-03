import numpy as np
from scipy.optimize import minimize as constraint_solve
from scipy.spatial.distance import pdist
from typing import Callable, Dict, NamedTuple, Optional, Tuple

Function = Callable[np.ndarray, float]


class LIPOSolution(NamedTuple):
    x: np.ndarray
    converged: bool
    n_evals: int
    
    def __str__(self) -> str:
        status = "yes" if self.converged else "no/unknown"
        return (
            f"x*: {self.x}\n"
            f"objective calls: {self.n_evals}\n"
            f"constraint calls: 0\n"
            f"B/B constraint calls: 0\n"
            f"converged: {status}"
        )


class LIPOSearch:
    
    def __init__(self, mfev: int, p: float=0.2, quasi_random: bool=False,
                 kvalues: Optional[np.ndarray]=None, max_sample_iters: int=100,
                 maxlipo: bool=True, maxlipo_starts: int=1,
                 maxlipo_method: Optional[str]=None, maxlipo_options: Optional[Dict]=None,
                 tr: bool=True, tr_max_pts: int=0, tr_max_radius: float=np.inf,
                 tr_method: Optional[str]=None, tr_options: Optional[Dict]=None,
                 verbose: bool=False) -> None:
        '''
        Creates a new instance of Max-LIPO-tr algorithm.
    
            Parameters:
                mfev (int): Maximum number of function evaluations
                p (float): Probability for exploration (random sampling).
                quasi_random (bool): Whether to replace uniform sampling with
                    quasi-random sampling (e.g. sample a point in the search
                    space with the largest L2 norm relative to points already
                    sampled)
                kvalues (iterable of int): Set of possible Lipschitz constants
                    (defaults to the sequence (1 + 0.01 * d) ** i)
                max_sample_iters (int): Maximum number of iterations to search
                    for an improvement point relative to the upper bound
                maxlipo (bool): Whether to maximize the upper bound
                maxlipo_starts (int): If maxlipo, how many restarts to use in
                    maximizing the upper bound
                maxlipo_method (string): Scipy optimizer to use for optimizing
                    the upper bound
                maxlipo_options (dict): Scipy options to pass to the optimizer
                    of the upper bound
                tr (bool): Whether to periodically perform local search by
                    fitting and optimizing a local quadratic model to the data
                tr_max_pts (int): Maximum number of points to use in fitting
                    the quadratic model (if not provided, uses the minimum 
                    required degrees of freedom 1 + n + n * (n + 1) / 2)
                tr_max_radius (float): Maximum radius of the trust region for 
                    the quadratic model
                tr_method (string): scipy optimizer to use for optimizing the 
                    local quadratic model
                tr_options (dict): Scipy options to pass to the optimizer of the
                    local quadratic model
                verbose (bool): Whether to print progress to the standard output
        '''
        self._mfev = mfev
        self._p = p
        self._quasi_random = quasi_random
        self._kvalues = kvalues
        self._verbose = verbose
        
        # Max-LIPO
        self._max_sample_iters = max_sample_iters
        self._maxlipo = maxlipo
        self._maxlipo_starts = maxlipo_starts
        self._maxlipo_method = maxlipo_method
        if maxlipo_options is None:
            maxlipo_options = {'maxiter': 1e6}
        self._maxlipo_options = maxlipo_options
        
        # trust region
        self._tr = tr
        self._tr_max_pts = tr_max_pts
        self._tr_max_radius = tr_max_radius
        self._tr_method = tr_method
        if tr_options is None:
            tr_options = {'maxiter': 1e6}
        self._tr_options = tr_options
    
    def initialize(self, f: Function, lower: np.ndarray, upper: np.ndarray,
                   guess: np.ndarray) -> None:
        '''
        Initializes the algorithm with the problem instance.
        
            Parameters:
                f (Callable): Function to optimize
                lower (ndarray): Lower box constraint of the search space
                upper (ndarray): Upper box constraint of the search space
                guess (ndarray): Initial guess for the search
        '''
        
        # assign problem
        self._n = lower.size
        self._f = f
        self._lower = lower
        self._upper = upper
        
        # initialize
        x1 = guess
        f1 = -self._f(x1)
        self._fev = 1
        self._xs = np.reshape(x1, (1, -1))
        self._fs = np.array([f1])
        self._khat = np.zeros(self._n)
        self._sigma = np.zeros(self._fs.size)
        self.hist = []
        self._it = 0
        if self._kvalues is None:
            self._kvalues = (1 + 0.01 * self._n) ** np.arange(1000)
            
    def iterate(self) -> None:
        '''
        Performs a single iteration of the current algorithm.
        '''
        
        trial_quad_solve = False
        trial_max_solve = False
        trial_sample_solve = False
        
        # exploration     
        if np.random.uniform() <= self._p:
            xn = self._sample_exploration_point()
                    
        # exploitation
        else: 
                        
            # even step: solve the trust region subproblem:
            #
            #    max_x f-hat(x) = max_x a + b' * x + x' * C * x
            #          s.t. lower[d] <= x[d] <= upper[d], for all d = 1...n
            #               || x - x_best || <= radius
            #       
            if self._tr and self._it % 2 == 0:
                xn = self._solve_quadratic_approximation()
                trial_quad_solve = xn is not None
            
            # odd step: maximize the upper bound U(x):
            #
            #    max_x U(x) = max_x min_{i=1...t} f_i + k * || x - x_i ||
            #          s.t. lower[d] <= x[d] <= upper[d], for all d = 1...n
            #
            if self._maxlipo and not trial_quad_solve:
                xn = self._maximize_upper_bound_function()
                trial_max_solve = xn is not None
            
            # fallback step: sample a potential maximizer by Lemma 8:
            #
            #    xn ~ { x : U(x) >= max_{i=1...t} f(x_i) }
            #        
            if not trial_quad_solve and not trial_max_solve:
                xn = self._sample_upper_bound_function()
                trial_sample_solve = xn is not None
                if not trial_sample_solve:
                    xn = self._sample_exploration_point()
                    
        # update memory and Lipschitz constant
        if not np.any(np.all(xn[np.newaxis] == self._xs, axis=1)):
            fn = -self._f(xn)
            self._fev += 1
            self._xs = np.vstack([self._xs, xn])
            self._fs = np.concatenate([self._fs, [fn]])
            self._khat, self._sigma = self._calculate_lipschitz_constant()
            
            # verbose
            bestf = -np.max(self._fs)
            self.hist.append(bestf)
            if self._verbose:
                lip = np.linalg.norm(self._khat)
                if trial_quad_solve:
                    method = 'quadratic'
                elif trial_max_solve:
                    method = 'max U'
                elif trial_sample_solve:
                    method = 'sample U'
                else:
                    method = 'random'
                print(f"it: {self._it} | fev: {self._fev} | obj: {bestf:.6f} | "
                      f"K: {lip:.6f} | method: {method}")
        self._it += 1
        
    def optimize(self, f: Function, lower: np.ndarray, upper: np.ndarray,
                 guess: np.ndarray) -> LIPOSolution:
        '''
        Runs the algorithm on the problem instance.
        
            Parameters:
                f (Callable): Function to optimize
                lower (ndarray): Lower box constraint of the search space
                upper (ndarray): Upper box constraint of the search space
                guess (ndarray): Initial guess for the search
            
            Returns:
                (LIPOSolution): The solution
        '''
        
        # start algorithm
        self.initialize(f, lower, upper, guess)
        
        # iterate
        converged = False
        while self._fev < self._mfev:
            self.iterate()
            if self._converged():
                converged = True
                break
        
        # extract the best solution found
        return LIPOSolution(
            x=self._xs[np.argmax(self._fs)],
            converged=converged,
            n_evals=self._fev
        )
    
    def solution(self) -> LIPOSolution:
        '''
        Returns the current best solution found.
        
            Returns:
                (LIPOSolution): The solution
        '''
        
        return LIPOSolution(
            x=self._xs[np.argmax(self._fs)],
            converged=False,
            n_evals=self._fev
        )
    
    def _converged(self) -> bool:
        return False
    
    # solve the optimization problem:
    #
    #    max_x min_{i=1...t} || x - x_i ||
    #        s.t. x in [lower, upper]
    #
    def _sample_exploration_point(self) -> np.ndarray:
        fun = lambda x:-np.min(np.sum((x[np.newaxis] - self._xs) ** 2, axis=1))        
        if self._quasi_random:
            result = constraint_solve(
                fun=fun,
                x0=np.random.uniform(self._lower, self._upper),
                bounds=list(zip(self._lower, self._upper))
            )
            x = result.x     
            u = np.random.uniform(self._lower, self._upper)
            return x if fun(x) < fun(u) else u
        else:
            return np.random.uniform(self._lower, self._upper)            

    # ==========================================================================
    # UPPER BOUNDING FUNCTION
    # ==========================================================================
    
    # calculates:
    #
    #    min { k[i] : max_{i != j} | f_i - f_i | / || x_i - x_j ||_2 <= k[i] }
    #
    def _calculate_lipschitz_constant(self) -> Tuple[np.ndarray, np.ndarray]:
        
        # estimate maximum slope
        xdist = pdist(self._xs)
        fdist = pdist(self._fs[:, np.newaxis])
        valid = xdist > 0
        max_slope = np.nanmax(fdist[valid] / xdist[valid])
        
        # estimate constant given partition
        ik = np.nonzero(self._kvalues >= max_slope)[0]  
        if ik.size > 0:
            khat = self._kvalues[ik[0]]
        else: 
            khat = max_slope        
        khat = np.full(shape=self._n, fill_value=khat)
        sigma = np.zeros(self._fs.size)
        return khat, sigma
    
    # calculates:
    #
    #    U(x) = min_i { f_i + sqrt(sigma_i + (x - x_i)' * K * (x - x_i)) }
    #
    def _upper_bound_function(self, x: np.ndarray) -> float:
        ks = self._khat[np.newaxis] ** 2
        norm2 = np.sum(ks * (x[np.newaxis] - self._xs) ** 2, axis=1)
        return np.min(self._fs + np.sqrt(self._sigma + norm2))
    
    # try to sample a point randomly from the set:
    #
    #    { x in [lower, upper] : U(x) > max_i f_i }
    #
    def _sample_upper_bound_function(self) -> Optional[np.ndarray]:
        fmax = np.max(self._fs)
        for _ in range(self._max_sample_iters):
            xn = np.random.uniform(self._lower, self._upper)
            if self._upper_bound_function(xn) > fmax:
                return xn
        return None

    # solve the optimization problem:
    #
    #    max_x U(x)
    #        s.t. x in [lower, upper]
    #
    def _maximize_upper_bound_function(self) -> Optional[np.ndarray]: 
        
        # for the starts, use the top k points as guesses
        if self._fs.size <= self._maxlipo_starts:
            return None
        topk = np.argsort(self._fs)[-self._maxlipo_starts:]
        
        # optimize U(x) starting from each guess until solution is an improvement
        fmax = np.max(self._fs)
        for idx in topk[::-1]:
            result = constraint_solve(
                fun=lambda x:-self._upper_bound_function(x),
                x0=self._xs[idx],
                bounds=list(zip(self._lower, self._upper)),
                method=self._maxlipo_method,
                options=self._maxlipo_options
            )
            x = result.x
            if self._upper_bound_function(x) > fmax:
                return x
        return None
    
    # ==========================================================================
    # QUADRATIC APPROXIMATION
    # ==========================================================================
    
    def _select_quadratic_interpolation_points(self) -> \
        Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[float]]:
        
        # check that there are enough points
        npts = 1 + self._n + (self._n * (self._n + 1)) // 2
        if self._fs.size < npts:
            return None, None
        npts = max(npts, self._tr_max_pts)
        
        # pick the points closest to the center within trmax
        xcenter = self._xs[np.argmax(self._fs)]
        norms = np.linalg.norm(self._xs - xcenter[np.newaxis], axis=1)
        xs = self._xs[norms <= self._tr_max_radius]
        fs = self._fs[norms <= self._tr_max_radius]
        norms = norms[norms <= self._tr_max_radius]
        idx = np.argsort(norms)[:npts]
        radius = np.max(norms[idx])
        return (xs[idx], fs[idx]), radius
    
    # compute the quadratic approximation by least squares:
    # 
    #    min_{a, b, C} sum_i ( f_i - f-hat(x_i; a, b, C) ) ^ 2
    #
    # where:
    # 
    #    f-hat(x; a, b, C) = a + b' * x + x' * C * x
    #
    def _build_quadratic_approximation(self) -> \
        Tuple[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], Optional[float]]:
        
        # design matrix for quadratic approximation around best point
        pts, radius = self._select_quadratic_interpolation_points()
        if pts is None:
            return None, None      
        xs, fs = pts  
        npts, n = xs.shape
        p = 1 + n + (n * (n + 1)) // 2
        features = np.ones((npts, p))
        features[:, 1:n + 1] = xs
        col = n + 1
        for i in range(n):
            for j in range(i, n):
                features[:, col] = xs[:, i] * xs[:, j]
                col += 1
                
        # find the least-squares parameters
        coeffs = np.linalg.lstsq(features, fs, rcond=None)[0]
        a = coeffs[0]
        b = coeffs[1:n + 1]
        C = np.zeros((n, n))
        col = n + 1
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    C[i, j] = coeffs[col]
                else:
                    C[i, j] = C[j, i] = coeffs[col] / 2
                col += 1
        return (a, b, C), radius
    
    # solve for the maximum of the quadratic:
    # 
    #     max_x f-hat(x; a, b, C)
    #         s.t. lower[d] <= x <= upper[d], for all d = 1...n
    #               || x - x0 || <= radius
    #
    def _solve_quadratic_approximation(self) -> Optional[np.ndarray]:
        
        # compute the quadratic approximation
        coeffs, radius = self._build_quadratic_approximation()
        if coeffs is None:
            return None
        a, b, C = coeffs
        
        # maximize the quadratic starting from the best solution so found
        x0 = self._xs[np.argmax(self._fs)]
        result = constraint_solve(
            fun=lambda x:-(a + np.dot(b, x) + np.dot(x, np.dot(C, x))),
            jac=lambda x:-(b + 2 * np.dot(C, x)),
            x0=x0,
            bounds=list(zip(self._lower, self._upper)),
            constraints=[
                { 
                    'type': 'ineq',
                    'fun': lambda x: radius ** 2 - np.sum((x - x0) ** 2),
                    'jac': lambda x:-2 * (x - x0)
                }
            ],
            method=self._tr_method,
            options=self._tr_options
        )
        return result.x
    
