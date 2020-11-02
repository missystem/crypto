"""
Authors: Missy Shi

Date: 05/22/2020

Python Version: 3.8.1

Functions: 
	- largest_prime_factor
	Find the largest prime factor of a number
	- prime_factors
	Given a integer, return a set of factors of the number
"""


import math


def largest_prime_factor(n: int) -> int:
    """ Return largest prime factor of number n """
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n


def prime_factors(n_1: int) -> set:
    """ Return a set of factors of number n_1 """
    lpf = largest_prime_factor(n_1)
    factors = set()
    factors.add(lpf)
    c = 16
    rest = n_1 // lpf
    while (rest != 1):
        lpf = largest_prime_factor(rest)
        rest = rest // lpf
        factors.add(lpf)
        c -= 1
    return factors