import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


# rs = RandomState(MT19937(SeedSequence(123456789)))

np.random.seed(42)

X = 5 * np.random.rand(100, 1)
y = 2 * X + 1 + np.random.rand(100, 1)

print("values of X:", X)
print("values of y:", y)