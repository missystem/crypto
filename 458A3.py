"""
Author: Missy Shi, Donna Hooshmand

Course: math 458

Date: 04/23/2020

Project: A3

Description:
Q1:
    Write a Python function which, given n,
    returns a list of all the primes less than n.
    There should be 25 primes less than 100, for instance.
    task: How many prime numbers are there which are less than 367400?
Q2:
    Implementation of Fast Powering Algorithm
    task: Compute the last five digits of the number 2 ** (10 ** 15)
Q3:
    Implementation of the Extended Euclidean Algorithm.
    task1. Find the inverse of 197189 (mod 999979),
    task2. Compute the numbers gcd(21000 − 229, 1000! + 98)
    task3. Find gcd(887519, 146744, 388025, 880464, 189074)
Q4:
    Implementation of the Miller-Rabin primality test
    using the fast algorithms.
    task: Use it to find a (probable) prime which is
          over 1000 bits long (≥ 302 digits).
Q5:
    Implementation of the Chinese Remainder theorem.

"""

import math
from functools import reduce
import random

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


# ===================================FINISHED IMPLEMENTATION=================================== #


def main():
    """ main program runner """

    print(f"{'-' * 45} Exercise 1 {'-' * 45}")
    n = 367400    # ex1
    num_primes = len(prime_num(n))
    print(f"{num_primes} prime numbers less than {n}.")
    print()

    print(f"{'-' * 45} Exercise 2 {'-' * 45}")
    n = 2
    p = 10 ** 15
    m = 100000
    print(f"{n}^{p} mod {m} = {fast_powering(n, p, m)}")
    print()

    print(f"{'-' * 45} Exercise 3 {'-' * 45}")
    # Find the inverse of 197189 (mod 999979) = 314159
    a = 197189
    b = 999979
    inv = mul_inverse(a, b)
    print(f"Modulo inverse of {a} (mod {b}) is {inv}")
    print()
    # Compute the numbers gcd(21000 − 229, 1000! + 98) = 1303
    a = (2 ** 1000 - 229)  # modulus
    b = math.factorial(1000) + 98
    a, b = a_bigger(a, b)  # helper function for determining which number is modulus
    g = gcd(a, b)
    print(f"gcd(21000 - 229, 1000! + 98) = {g}")
    print()
    # Find gcd(887519, 146744, 388025, 880464, 189074) = 1411
    num = [887519, 146744, 388025, 880464, 189074]
    g = gcd_of_list(num)
    print(f"gcd(887519, 146744, 388025, 880464, 189074) = {g}")
    print()

    print(f"{'-' * 45} Exercise 4 {'-' * 45}")
    while 1:
        n = random.randint(10 ** 302, 10 ** 350)  # find a prime which is over 1000 bits long
        rhypo = int(2 * (math.log(n)) ** 6)
        a = random.randint(2, rhypo)  # choose random integer as witness
        if miller_rabin_test(n, a) == False:
            print(f"Tests failed: {n} is a (probable) prime, tested with candidate witness: {a}")
            break
    print()

    print(f"{'-' * 45} Exercise 5 {'-' * 45}")
    # small example for testing chinese remainder theorem
    n = [3, 7, 16]
    a = [2, 3, 4]
    x = crt(n, a)
    print(f"list of numbers x1, ..., xk: {a} \nlist of moduli m1, ..., mk: {n} \n=> x = {x}")
    print()


    print(f"{'-' * 45} Exercise 6 {'-' * 45}")
    # p = 37055228012567588205472561716198899109643     # p is 41 digits integer
    p = 10**200 + 357
    g = primitive_root(p)
    h = 100  # log g (100) (mod p) <=> g^x = 100 (mod p)

    # Part (a) Why p is probably prime?
    print("Part(a): ")
    print("    If p is composite, then at least 75% of the numbers a between 1 and p-1 are Miller-Rabin \n    "
          "witnesses for n. By AKS Primality Test (p137 on book), we are able to determine whether  \n    "
          "a given number p is prime in no more than O((ln(p))^(6)) steps.\n    "
          "Thus, I had param: akspt to choose candidate witness a from random integer between\n    "
          "1 and (ln(p))^(6) for 50 times, which means we test number p with 50 different a, \n    "
          "if didn't fail, it's conclusively high probability that p is a prime number.\n")
    num_wit = 50
    print(f"Use Miller-Rabin test to check for [{p}] {num_wit} times:")
    mrt_random_a(p, num_wit)
    print()

    # Part (b)
    print("Part(b)")
    print(f"Smallest primitive root {g} mod {p} is: {primitive_root(p)}")
    print()

    # Part (c): Why is it practical to compute log g (100) (mod p)?
    print("Part(c): ")
    print("    p has approximately 41 decimal digits (2^136 bits), which is not a very large number\n    "
          "for computer to compute, and an element g whose order is prime and approximately p/2,\n    "
          "then it is practical to compute log g (100) (mod p)")
    print()


    # Part (d)
    print("Part(d)")
    x = primitive_ph(g, h, p)
    print(f"Solve for {g}^x = {h} mod {p} \n  => x = {x}")
    print()
    print(f"Check if {x} is the correct answer...")
    if x == 22945092366571147534189420205656207566200:
        print("Yes, it is correct.")
    else:
        print("No, it is not correct.")

    print(f"\n{'-' * 45} DONE {'-' * 45}")



if __name__ == '__main__':
    main()

