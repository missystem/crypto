"""
Author: Missy Shi

Course: math 458

Date: 04/23/2020

Project: A3 - 1

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
# from numpy import log as ln
import random

# ================================Ex 1================================ #


def is_prime(n: int) -> bool:
    """ Determine given integer is prime or not,
        return True if it is a prime,
        return False otherwise """
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


def prime_num(n: int) -> list:
    """ Find primes less than given number n """
    pl = []
    for i in range(1, n):
        if is_prime(i) is True:
            pl.append(i)
    return pl


# ================================Ex 2================================ #


def bin_expansion(p: int) -> int:
    """ find binary expansions from given exponent p """

    count = 0
    while p > 1:
        count += 1
        p = p - (2**count)
    return count


def fast_pow(n, p, m):
    """ Compute n ** p (mod m) """

    cl = []  # list of binary expansions
    while p > 0:
        count = bin_expansion(p)
        p = p - 2**count
        cl.append(count)

    pl = []  # list of n ** binary expansions
    for c in cl:
        pl.append((n**(2**c)) % m)

    result = 1
    for i in pl:
        result *= i

#     print(result % m)
    return result % m


# ================================Ex 3================================ #


def gcd(a: int, b: int) -> int:
    """ Find gcd """

    if a == 0 or b == 0:
        return max(a, b)

    if b > a:
        temp = a
        a = b
        b = temp

    q = []  # list of quotients
    r = []  # list of remainders
    while b != 0:
        temp = b
        q.append(a // b)
        b = a % b
        r.append(b)
        a = temp
    return a


def gcdEx(a, b) -> (int, int):
    """ Extended Euclidean Algorithm """
    # a, b = a_bigger(a, b)
    p0, p1 = 0, 1
    q0, q1 = 1, 0
    listP = [0, 1]
    listQ = [1, 0]

    while b != 0:
        q = a // b
        a, b = b, a % b
        p0, p1 = p1 - q * p0, p0
        listP.append(p0)
        q0, q1 = q1 - q * q0, q0
        listQ.append(q0)

    # print(listP)
    # print(listQ)
    u = p1
    v = q1
    # print(u, v)
    return u, abs(v)


def a_bigger(a, b):
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
    if b == 1:
        return b
    u, v = gcdEx(a, b)
    return u


# ================================Ex 4================================ #


def miller_rabin_test(n, a):
    """ Return True if some witness a proves that n is composite,
        Return False otherwise """

    witness = a
    if n % 2 == 0 or gcd(n, a) != 1 or gcd(n, 2310) != 1:
        print(f"{n} is a composite, {witness} is a potential witness")
        return True

    k = 0
    q = n - 1
    while q % 2 == 0:
        q = q // 2
        k += 1

#     a = (a**q) % n
    #WE ARE USING THE FAST POWERING ALG.
    a = fast_pow(a, q, n)
    if (a % n == 1) or (a % n == (n - 1)):
        # print(f"Test Fails. {n} is a probable prime")
        return False

    for i in range(0, k - 1):
        a = (a**2) % n
        if a % n == (n - 1):
            # print(f"Test Fails. {n} is a probable prime")
            return False
        # print(a)
    print(f"{n} is a composite, {witness} is a potential witness")
    return True


def q6a(n):

    akspt = int(2 * (math.log(n))**6)
    num_wit = 50
    for i in range(num_wit):
        a = random.randint(1, akspt)  # choose random integer as witness
        # print(f"chosen random int {a} as candidate witness")
        if miller_rabin_test(n, a) is True:
            return
    print(
        f"Tests failed: {n} is a(probable) prime after test with {num_wit} different witness."
    )
    """ If p is composite, then at least 75% of the numbers a between 1 and p-1
        are Miller-Rabin witnesses for n.
        Through AKS Primality Test, we can know that we are able to determine
        whether a given number p is prime in no more than O((ln(p))^(6)) steps
        Thus, I had param: akspt to choose candidate witness a from random
        integer between 1 and (ln(p))^(6) for 50 times, which means we test
        number p with 50 different a, if didn't fail, it's conclusively
        high probability that p is a prime number.
    """


# ==================================================================== #


def crt(n: list, a: list) -> int:

    g = gcd_of_list(n)
    if g != 1:
        print(f"gcd({n}) != 1")
        return
    sum = 0
    N = reduce(lambda a, b: a * b, n)
    # print(N)
    for i in range(len(n)):
        n_i = n[i]
        a_i = a[i]
        # print(n_i, a_i)
        b_i = N // n_i

        # TODO  TODO  TODO
        inv, v = gcdEx(b_i, n_i)
        # print(v)
        sum += a_i * inv * b_i
    result = sum % N
    # print(result)
    return result


def q6b(n):
    ord = []  # possible orders
    temp = n - 1
    # ord.append(temp)
    # while temp % 2 == 0:
    for i in range(2, temp - 1):
        if (temp % i) == 0:
            ord.append(i)
    # print(ord)
    for i in range(1, n - 1):
        for j in ord:
            if (i**j) % n == 1:
                break
            if j == ord[-1] and (i**j) % n != 1:
                return i
            
            
            
def example93():
    g = 5448
    h = 6909
    # p = 11251
    p = 37055228012567588205472561716198899109643
    p_1 = p - 1
    d = {}
    for i in range(2, p - 1):
        if is_prime(i):
            while p_1 % i == 0:
                if i in d:
                    d[i] += 1
                else:
                    d[i] = 1
                p_1 = p_1 // i
#     print(d)
    key = value = 0
#     print(d.items())
    for k, v in sorted(d.items()):
#         print(k, v)
        if fast_pow(g, k**v, p) == 1:
            key += k
            value += v-1
            break
#     print(key, value)
    xlist = []
    value_copy = value
    x = 0
    invofg = mul_inverse(g, p)
#     print(invofg)
    gtothekeypowervalue = fast_pow(g, key ** value, p)
#     print(f"gtothekeypowervalue when i == {i}: {gtothekeypowervalue}")
    for i in range(value_copy+1):
        if i == 0:
            htothekeypowervalue = fast_pow(h, key ** value, p)
            print(f"htothekeypowervalue when i == 0: {htothekeypowervalue}")
        else:
            htothekeypowervalue = fast_pow((h * invofg), i * x * key ** value, p)
            print(f"htothekeypowervalue when i == {i}: {htothekeypowervalue}")
        
            # (g^key^value)^x = h^key^value
        for j in range(value_copy+1):
            print(j)
            if fast_pow(gtothekeypowervalue, j, p) == htothekeypowervalue:
                xlist.append(j)
                x += j * key ** i
                print(f"x{i} == {x}")

        value -= 1
#     print(xlist)

        
            


def main():
    
    example93()
    print()
    print()
    
    
    """ main program runner """
    """
    print(f"{'-' * 25} Exercise 1 {'-' * 25}")
    # n = 367400    # ex1
    n = 200  # small example: # of primes less than 200: 46
    prime_list = prime_num(n)
    num_primes = len(prime_list)
    print(f"{num_primes} prime numbers less than {n}:\n{prime_list}")

    print(f"\n{'-' * 25} Exercise 2 {'-' * 25}")
    # n = 2
    # p = 10 ** 15
    # m = 100000
    n, p, m = 2, 20, 100  # small example: 2^20 mod 100 = 76
    print(fast_pow(n, p, m))
    

    print(f"\n{'-' * 25} Exercise 3 {'-' * 25}")
    # Find the inverse of 197189 (mod 999979) = 314159
    a = 197189
    b = 999979
    inv = mul_inverse(a, b)
    print(f"inverse of {a} (mod {b}) is {inv}")

    # Compute the numbers gcd(21000 − 229, 1000! + 98) = 1303
    a = (2**1000 - 229)  # modulus
    b = math.factorial(1000) + 98
    a, b = a_bigger(
        a, b)  # helper function for determining which number is modulus
    g = gcd(a, b)
    print(f"gcd(21000 - 229, 1000! + 98) = {g}")

    # Find gcd(887519, 146744, 388025, 880464, 189074) = 1411
    num = [887519, 146744, 388025, 880464, 189074]
    g = gcd_of_list(num)
    print(f"gcd(887519, 146744, 388025, 880464, 189074) = {g}")

    print(f"\n{'-' * 25} Exercise 4 {'-' * 25}")
    # n = 122430513841  # 5_000_000_000th prime
    # n = random.randint(10 ** 302, 10 ** 350)  # find a prime which is over 1000 bits long
    n = 3571  # small example: 500th prime
    rhypo = int(2 * (math.log(n))**6)
    a = random.randint(2, rhypo)  # choose random integer as witness
    if miller_rabin_test(n, a) == False:
        print(
            f"Tests failed: {n} is a (probable) prime test with candidate witness: {a}"
        )

    print(f"\n{'-' * 25} Exercise 5 {'-' * 25}")
    n = [3, 7, 16]
    a = [2, 3, 4]
    x = crt(n, a)
    print(
        f"list of numbers x1, ..., xk: {a} \nlist of moduli m1, ..., mk: {n} \n=> x = {x}"
    )

    print(f"\n{'-' * 25} Exercise 6 {'-' * 25}")
    # p = 37055228012567588205472561716198899109643     // p is prime
    # p is 10^41 chars
    p = 3571  # 500th prime
    # p = 3572    # not prime
    # Part (a)
    q6a(p)
    print()
    """
    """ Part (b) """
#     min_pr = q6b(p)
#     print(f"{min_pr} is the smallest primitive root of {p}")
#     print()
    
    # min_pr ** x = 100 (mod p)

    # Part (c)
    """ Part(c): Why is it practical to compute log g (100) (mod p)?
        A: p has approximately 41 decimal digits (2^136 bits), which
        is not a very large number to compute,
        and an element g whose order is prime and approximately
        p/2, then it is practical to compute log g (100) (mod p) """

    # a = 336
    # b = 291
    # u, v = gcdEx(a, b)
    # g = gcd(a, b)
    # print(f"{a} * {u} - {b} * {v} = {g}")
    

if __name__ == '__main__':
    main()
