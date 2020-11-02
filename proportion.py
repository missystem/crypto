"""
Author: Missy Shi

Course: math 458

Date: 05/06/2020

Project: find a prime number for ph algorithm run fast

"""

import math
from functools import reduce
import random
import time

# ================================Ex 1================================ #


def prime_num(n: int) -> list:
    """ Returns  a list of primes < n """
    sieve = [True] * (n // 2)
    for i in range(3, int(n ** 0.5) + 1, 2):
        if sieve[i // 2]:
            sieve[i * i // 2::i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
    return [2] + [2 * i + 1 for i in range(1, n // 2) if sieve[i]]


# ================================Ex 2================================ #


def bin_expansion(p: int) -> int:
    """ find binary expansions from given exponent p """
    count = 0
    while p > 1:
        count += 1
        p = p - (2 ** count)
    return count


def fast_powering(n, pow, modulus):
    res = 1
    n = n % modulus
    while pow > 0:
        if int(pow) & 1:
            res = (res * n) % modulus
        pow = int(pow) >> 1
        n = (n * n) % modulus
    return res


# ================================Ex 3================================ #


def gcd(a: int, b: int) -> int:
    """ Return gcd(a, b) """
    if a == 0 or b == 0:
        return max(a, b)

    if b > a:
        temp = a
        a = b
        b = temp

    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a


def gcdEx(a, b) -> (int, int):
    """ Extended Euclidean Algorithm """
    p0, p1 = 0, 1
    q0, q1 = 1, 0

    while b != 0:
        q = a // b
        a, b = b, a % b
        p0, p1 = p1 - q * p0, p0
        q0, q1 = q1 - q * q0, q0

    u = p1
    v = q1
    return u, abs(v)


def a_bigger(a: int, b: int) -> (int, int):
    """helper function, determine which int should be modulus """
    if b > a:
        temp = a
        a = b
        b = temp
    return a, b


def gcd_of_list(num: list) -> int:
    """ Return gcd of a list of numbers """
    num1 = num[0]
    num2 = num[1]
    g = gcd(num1, num2)
    if len(num) > 2:
        for i in range(2, len(num)):
            g = gcd(g, num[i])
    return g


def mul_inverse(a, b):
    """ Find Multiplicative Inverse by using extended euclidean algorithm """
    if b == 1:
        return b
    u, v = gcdEx(a, b)
    return u


# ================================Ex 4================================ #


def miller_rabin_test(n: int, a: int) -> bool:
    """ Miller-Rabin Primality Test

    :param n: positive integer to test
    :param a: possible witness (integer < n)
    :return: True if a witness proves that n is composite, False otherwise
    """

    witness = a
    if n % 2 == 0 or gcd(n, a) != 1 or gcd(n, 2310) != 1:
        return True

    k = 0
    q = n - 1
    while q % 2 == 0:
        q = q // 2
        k += 1
    a = fast_powering(a, q, n)
    if (a % n == 1) or (a % n == (n - 1)):
        return False

    for i in range(0, k - 1):
        a = (a ** 2) % n
        if a % n == (n - 1):
            return False
    return True


def mrt_random_a(n: int, num_wit: int) -> None:
    """ miller-rabin test with num_wit random a
    :param n:
    :param num_wit:
    :return: None
    """
    akspt = int(2 * (math.log(n)) ** 6)
    wit = num_wit
    while num_wit != 0:
        a = random.randint(1, akspt)  # choose random integer as witness
        if miller_rabin_test(n, a) is True:
            return
        num_wit -= 1
    print(f"Tests failed:\n   {n} is a (probable) prime after test with {wit} different witness.")


# ==================================================================== #


def crt(n: list, a: list) -> int:
    """ Chinese Remainder Theorem Implementation """
    g = gcd_of_list(n)
    if g != 1:
        print(f"gcd({n}) != 1")
        return -1

    sum = 0
    N = reduce(lambda a, b: a * b, n)
    for i in range(len(n)):
        n_i = n[i]
        a_i = a[i]
        b_i = N // n_i
        inv, v = gcdEx(b_i, n_i)
        sum += a_i * inv * b_i
    result = sum % N
    return result


def largest_prime_factor(n):
    """ Return largest prime factor of number n """
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n


def prime_factors(n_1):
    """ Return a set of factors of number n_1 """
    lpf = largest_prime_factor(n_1)
    factors = set()
    factors.add(lpf)
    c = 16
    rest = n_1 // lpf
    while(rest != 1):
        lpf = largest_prime_factor(rest)
        rest = rest // lpf
        factors.add(lpf)
        c -= 1
    return factors


def occurrence(n_1):
    """ Return a dictionary of occurrence of a set of number {num: occurrence} """
    factors = prime_factors(n_1)
    occur = {}
    for i in factors:
        occur[i] = 0
        while n_1 % i == 0:
            occur[i] += 1
            n_1 = n_1 // i
    return occur


def primitive_root(n: int) -> int:
    """
    Return smallest primitive root of integer n
    :param n: prime number
    :return: r if found primitive root, -1 if no result
    """
    n_1 = n - 1
    order = prime_factors(n_1)  # set of possible orders

    for r in range(2, n):
        flag = False
        for it in order:
            if fast_powering(r, n_1 // it, n) == 1:
                flag = True
                break
        if flag is False:
            return r
    return -1


def primitive_ph(g: int, h: int, p: int) -> int:
    """ Pohlig-Hellman Algorithm for g is primitive root """

    occur = occurrence(p-1)
    a = []
    n = []
    for q, e in sorted(occur.items()):
        gpowqe = fast_powering(g, (p - 1) // q ** e, p)
        hpowqe = fast_powering(h, (p - 1) // q ** e, p)
        i = 0
        while fast_powering(gpowqe, i, p) != hpowqe:
            i += 1
        x = i

        a.append(x)
        n.append(q ** e)
    res = crt(n, a)
    return res





def findPrimeNumber(r):
    """r: range"""
    while 1:
        n = random.randint(2 ** (r-1), 2 ** (r+1))
        rhypo = int(2 * (math.log(n)) ** 6)
        a = random.randint(2, rhypo)  # choose random integer as witness
        if miller_rabin_test(n, a) == False:
            print(f"Tests failed: {n} is a (probable) prime, tested with candidate witness: {a}")
            break
    return n


def prop(r):
    n = 2
    l = []
    while True:
        phi = 1
        for i in range(2, n):
            if gcd(i, n) == 1:
                phi += 1
        if (phi/n).__round__(5) == r.__round__(5):
            l.append(n)
            if len(l) == 1:
                break
        else:
            n += 1
    return l



# ===================================FINISHED IMPLEMENTATION=================================== #


def main():
    """ main program runner """
    r = 4/15
    print(f"P({prop(r)}) = {r}")





if __name__ == '__main__':
    main()

