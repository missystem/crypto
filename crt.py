"""
Author: Missy Shi

Course: math 458

Date: 04/23/2020

Project: A3 - 4

Description:
    Implement the Chinese Remainder theorem in Python.
    That is, given a list of numbers x1,...,xk and
    a list of moduli m1,...,mk, solve the system of congruences
            x ≡ x1 (mod m1)
                ...
            x ≡ xk (mod mk)
Notes:
    Make sure you’re using the fast algorithms you implemented
    in exercises 2 and 3 to do this, so that you can handle
    big numbers. You can do this the way I explained in class,
    but it’s probably easier if you do what it says on the wikipedia
    entry for the Chinese remainder theorem, in the “Existence
    (direct construction)” section.
"""
from functools import reduce


def chinese_remainder(n, a):
    # sum = 0
    # prod = reduce(lambda a, b: a*b, n)
    # # print(prod)
    # for n_i, a_i in zip(n,a):
    #     p = prod // n_i
    #     sum += a_i* mul_inv(p, n_i)*p
    # return sum % prod
    x = 0
    # N = reduce(lambda ni, nj: ni * nj, n)

    # m1 = n[0]
    # m2 = n[1]
    # a1 = a[0]
    # a2 = a[1]
    # y = (mul_inv(m1, m2) * (a2 - a1)) % m2
    # x = a1 + m1 * y
    # m3 = n[2]
    # a3 = a[2]
    # tempM = (m1 * m2) % m3
    # tempA = (a3 - x) % m3
    # z = (mul_inv(tempM, m3) * tempA) % m3
    # x += m1 * m2 * z
    # print(x)
    for i in range(len(n)-1):
        mi = n[i]
        mj = n[i+1]
        ai = a[i]
        aj = a[i+1]
        temp = (mul_inv(mi, mj) * (aj - ai)) % mj
        x +=



def mul_inv(a, b):
    b0 = b
    x0, x1 = 0, 1
    if b == 1:
        return 1
    while a > 1:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += b0
    return x1


def main():
    """ main program runner """
    # n = [3, 5, 7]
    # a = [2, 3, 2]
    n = [3, 7, 16]
    a = [2, 3, 4]
    chinese_remainder(n, a)
    # print(chinese_remainder(n, a))


if __name__ == '__main__':
    main()