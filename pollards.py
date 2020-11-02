"""
Author: Missy Shi
Description: Pollard's p-1 Factorization Algorithm
"""


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


def fast_powering(n, pow, modulus):
    res = 1
    n = n % modulus
    while pow > 0:
        if int(pow) & 1:
            res = (res * n) % modulus
        pow = int(pow) >> 1
        n = (n * n) % modulus
    return res


def pollard(N):
    a = 2
    j = 2
    while True:
        a = fast_powering(a, j, N)
        d = gcd(a - 1, N)
        if 1 < d < N:
            return d
        j += 1


def main():
    N = 13927189
    p = pollard(N)
    print(f"N = {N}, p = {p}, q = {N//p}")

    N = 5429
    p = pollard(N)
    print(f"N = {N}, p = {p}, q = {N // p}")

if __name__ == '__main__':
    main()