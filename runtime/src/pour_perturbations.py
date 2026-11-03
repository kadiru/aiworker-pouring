"""Small deterministic motor disturbances for corrective-label demonstrations."""
from functools import lru_cache
import numpy as np


def smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*(3-2*x)


@lru_cache(maxsize=32)
def profile(seed):
    rng=np.random.default_rng(seed)
    return rng.uniform(1.1,1.9,7),rng.uniform(0,2*np.pi,7)


def approach_noise(t,seed,amplitude_deg):
    frequency,phase=profile(seed)
    gate=smooth((t-.15)/.25)*(1-smooth((t-2.1)/.3))
    weights=np.array([1.,1.,.5,.7,.7,.6,.5])
    return amplitude_deg*weights*gate*np.sin(2*np.pi*frequency*t+phase)
