"""
Author: Missy Shi

Course: math 458

Date: 04/23/2020

Project: A3 - 2

Description:
    Implementation of Fast Powering Algorithm

Task:
    Compute the last five digits of the number 2 ** (10 ** 15)
"""


def bin_expansion(p: int) -> int:
    """ find binary expansions from given exponent p """

    count = 0
    while p > 1:
        count += 1
        p = p - (2 ** count)
    return count


def mod(num: int, m: int) -> int:
    """ compute given number(num) mod modulus(m) """
    while num - m > 0:
        num -= m
    return num


def q2():
    """ Compute n ** p (mod m) """
    n = 2
    p = 10 ** 15
    m = 100000

    cl = []  # list of binary expansions
    while p > 0:
        count = bin_expansion(p)
        p = p - 2 ** count
        cl.append(count)

    pl = []  # list of n ** binary expansions
    for c in cl:
        pl.append(mod(n ** (2 ** c), m))

    result = 1
    for i in pl:
        result *= i

    print(mod(result, m))
    return


def main():
    """ main program runner """
    q2()


if __name__ == '__main__':
    main()