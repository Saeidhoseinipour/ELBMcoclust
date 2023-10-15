#!/usr/bin/env python
# coding: utf-8

# In[ ]:



"""
Module `ELBMcoclust.Models` module gathers implementations of co-clustering
Modified CEM Algorithms with (Sparse) Exponential  Family Latent Block Model.
"""

from .coclust_ELBMcem import CoclustELBMcem
from .coclust_SELBMcem import CoclustSELBMcem



__all__ = ['CoclustELBMcem',
'CoclustSELBMcem']
