#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:44:05 2021

@author:
"""
from math import factorial
import math
import numpy as np
n = 14 * 14
def c(n,i):
    return factorial(n) // ( factorial(i)*factorial(n-i) )

x = 1+sum([c(n,i)*c(n-i+1,i) for i in range(1,(n+3)//2)])

print("Possible Combinations My Solution", x < 3**196)
print(3**196)
print(3**196 / 10**93)
print(3.27 * 10**93)

