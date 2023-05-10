#%%
import numpy as np
import pandas as pd
#%%
a = [[1, 2], [3, 4], [5, 6]]
b = [x[0] for x in a]

b = []
for x in a:
    for _ in range(x[1]):
        b += [x[0]]

#In list interpretation form
b = [x[0] for x in a for _ in range(x[1])]
# %%
