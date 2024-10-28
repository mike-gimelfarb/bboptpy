import numpy as np
from bboptpy import ActiveCMAES


# function to optimize
def fx(x):
    total = 0.0
    for x1, x2 in zip(x[:-1], x[1:]):
        total += 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    return total


n = 10  # dimension of problem
alg = ActiveCMAES(mfev=10000, tol=1e-4, np=20)
alg.initialize(f=fx,
               lower=-10 * np.ones(n),
               upper=10 * np.ones(n),
               guess=np.random.uniform(low=-10., high=10., size=n))
for _ in range(500): 
    alg.iterate()
    print(alg.solution())

    
