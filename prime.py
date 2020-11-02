"""
Author: Missy Shi

Course: math 458

Date: 04/23/2020

Project: A3 - 1

Description:
    Write a Python function which, given n,
    returns a list of all the primes less than n.
    There should be 25 primes less than 100, for instance.

Task:
    How many prime numbers are there which are less than 367400?
"""

import math


def is_prime(n: int) -> bool:
    """ Check if a integer is prime,
    If it is, return True, else return False """
    if n < 2:
        # print(f"{n} is not a prime number")
        return False
    else:
        for i in range(2, n):
            if (n % i) == 0:
                # print(f"{n} is not a prime number")
                # print(f"{n} divides {i} = {n//i}")
                return False
        return True


def q1():
    """ Find primes less than given number n """
    # n = int(input('Input an integer: '))
    n = 367400
    pl = []
    for i in range(1, n):
        if is_prime(i) is True:
            pl.append(i)
    count = len(pl)
    print(f"{count} prime numbers less than {n}")
    # print(pl)
    return


def main():
    """main program runner"""
    q1()


if __name__ == '__main__':
    main()