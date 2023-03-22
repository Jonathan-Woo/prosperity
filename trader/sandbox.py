#%%
import numpy as np
import pandas as pd
#%%
a = {
    "a": 1,
    "b": 2,
    "c": 3
}

b = sorted(list(a.items()), key=lambda x: x[0], reverse=True)
# %%
