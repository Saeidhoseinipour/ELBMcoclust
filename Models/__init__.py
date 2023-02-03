#!/usr/bin/env python
# coding: utf-8

# In[ ]:



"""
Module `ELBMcoclust.Models` module gathers implementations of co-clustering
Modified EM Algorithms with Exponential  Family Latent Block Model.
"""

from .coclust_ELBMcem import CoclustELBMcem
#from .coclust_SELBMcem_v2 import CoclustELBMcem
from .coclust_SELBMcem_v3 import CoclustSELBMcem



__all__ = ['CoclustELBMcem',
'coclust_plbcem',
'CoclustSELBMcem']