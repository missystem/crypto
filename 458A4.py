"""
Authors: Missy Shi, Donna Hooshmand
Group Name: Space is Big

Description: Implementations of Sieve (find primes < N) and Gaussian Elimination

Course: MATH458

Instructor: Prof. Ben Young

Date: 05/14/2020

Notes:
    1) Find the sum of all prime numbers less than N, where N is as big as you can.
    Bigger values of N get more points, loosely speaking.
    Basically I'm going to do this assignment myself in several different ways
    (one inefficient, one pretty good, and one good)
    and see how far in the computation I can get in 10 minutes
    or so using each of the algorithms

    2) Write code to solve a system of linear equations modulo p.
    I'll give you an explicit system to solve.

Credits:
    https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n
"""

# ------------------------------------- Function Declaration ------------------------------------- #
# prime_num(n: int) -> list
# sieve_of_eratosthenes(n: int) -> list
# gcd(a: int, b: int) -> int
# gcdEx(a: int, b: int) -> (int, int, int)
# mul_inverse(a: int, m: int) -> int
# fast_powering(n: int, pow: int, modulus: int) -> int
# swap(matrix: list, i: int, j: int)
# gaussian(mtx: list) -> list
# find_free(mtx: list) -> int
# find_x(M: list) -> list
# ------------------------------------------------------------------------------------------------ #

from functools import reduce
import random
import time


def prime_num(n: int) -> list:
    """ Returns  a list of primes < n """
    sieve = [True] * n
    for i in range(3, int(n ** 0.5) + 1, 2):
        if sieve[i]:
            sieve[i * i::2 * i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
    return [2] + [i for i in range(3, n, 2) if sieve[i]]


def sieve_of_eratosthenes(n: int) -> list:
    """return the list of the primes < n"""
    if n <= 2:
        return []
    sieve = range(3, n, 2)
    top = len(sieve)
    for si in sieve:
        if si:
            bottom = (si*si - 3) // 2
            if bottom >= top:
                break
            sieve[bottom::si] = [0] * -((bottom - top) // si)
    return [2] + [el for el in sieve if el]


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


def gcdEx(a: int, b: int) -> (int, int, int):
    """ Extended Euclidean Algorithm """
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = gcdEx(b % a, a)
        return (g, x - (b // a) * y, y)


def mul_inverse(a: int, m: int) -> int:
    """
    Find the multiplicative inverse of a mod m
    :param a: integer
    :param m: modulus
    :return: integer a^(-1) (mod m)
    """
    g, x, y = gcdEx(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m


def fast_powering(n: int, pow: int, modulus: int) -> int:
    """ Implementation of fast powering algorithm
    :param n: base integer
    :param pow: exponent
    :param modulus: integer modulus
    :return: res = n**pow (mod modulus)
    """
    res = 1
    n = n % modulus
    while pow > 0:
        if int(pow) & 1:
            res = (res * n) % modulus
        pow = int(pow) >> 1
        n = (n * n) % modulus
    return res


def swap(matrix: list, i: int, j: int):
    """ a function that switchs rows i and j
    :param matrix: the given matrix we want to apply these operatios on
    :param i: row i that we will be switching
    :param j: row j that we will be switching
    :return: the new matrix with the swapped rows
    """
    temp = matrix[i]
    matrix[i] = matrix[j]
    matrix[j] = temp
    return matrix


def gaussian(mtx: list) -> list:
    """ This function will do the gaussian elimination algorithm to get the row reduced matrix
    :param mtx: the matrix we wish to solve
    :return: the row reduced matrix
    """
    num_row = len(mtx)
    num_col = len(mtx[0])
    # while True:
    pivot = 0
    while True:
        # print(f"{pivot+1}-th round: ")
        for i in range(pivot, num_row):
            if mtx[i][pivot] == 1:
                mtx = swap(mtx, pivot, i)
                break
        for i in range(pivot+1, num_row):
            if mtx[i][pivot] == 1:
                for j in range(pivot, num_col):
                    mtx[i][j] = (mtx[i][j] + mtx[pivot][j]) % 2
        # for i in range(num_row):
        #     print(mtx[i])
        # print()
        pivot += 1
        if pivot == num_row:
            return mtx


def find_free(mtx: list) -> int:
    """ This function finds the free variables of the matrix
    :param mtx: This will be the row reduced matrix we get from gaussian()
    :return: index of the free variable
    """
    free = [0 * i for i in range(len(mtx[0]))]

    if free not in mtx:
        return -1
    return mtx.index(free)


def find_x(M: list) -> list:
    """ This function will find the vector/solution to the equation Mx=0, where M is the matrix
    :param M: The matrix we wish to solve
    :return: the vector x, which is the non-trivial solution for the equation Mx=0
    """
    mtx = gaussian(M)
    num_row = len(mtx)
    num_col = len(mtx[0])

    if find_free(mtx) == -1:
        print(f"There is no free variable in the matrix, we only have trivial solution")
        return [0 * i for i in range(num_col)]

    x = []
    for r in range(num_row):
        x.append(0)
    count = num_row
    list_of_index_of_0 = []
    while count > 0:
        index_of_0 = 0
        for i in range(num_row):
            if sum(mtx[i]) == 1:
                index_of_0 += mtx[i].index(1)
                list_of_index_of_0.append(index_of_0)
                break
        for j in range(num_row):
            if index_of_0 != j:
                mtx[j][index_of_0] = 0
        count -= 1

    for k in range(num_row):
        x[k] = (sum(mtx[k]) - 1) % 2
    for ex in list_of_index_of_0:
        x[ex] = 0
    return x


# ------------------------------------------------- #


def main():
    print(f"Question 1: ")
    r = 30
    n = random.randint(2 ** (r - 1), 2 ** (r + 1))
    print(f"The random generated integer N is {n}")
    time1 = time.perf_counter()
    l = prime_num(n)
    time2 = time.perf_counter()
    s = reduce(lambda x, y: x + y, l)
    time3 = time.perf_counter()
    print(f"Sum of the prime number less than N is {s}")
    print(f"Time spent on finding prime number less than {n} is {(time2-time1).__round__(3)} \nTime spent on summing up is {(time3-time2).__round__(3)}")

    # (1)
    # The random generated integer N is 846893546
    # Sum of the prime number less than N is 17891196676494016
    # Time spent on finding prime number less than 846893546 is 46.203
    # Time spent on summing up is 3.907
    # (2)
    # The random generated integer N is 897664202
    # Sum of the prime number less than N is 20041605020899270
    # Time spent on finding prime number less than 897664202 is 49.734
    # Time spent on summing up is 4.19
    # (3)
    # The random generated integer N is 1000294057
    # Sum of the prime number less than N is 24753695180736318
    # Time spent on finding prime number less than 1000294057 is 56.003
    # Time spent on summing up is 4.662
    # (4)
    # The random generated integer N is 1599950544
    # Sum of the prime number less than N is 61889567968539089
    # Time spent on finding prime number is 93.226
    # Time spent on summing up is 7.252
    # (5)
    # The random generated integer N is 2738313974
    # Sum of the prime number less than N is 176694249436555500
    # Time spent on finding prime number less than 2738313974 is 201.544
    # Time spent on summing up is 12.985

    print(f"\nQuestion 2:")
    A = [[0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
         [0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
         [1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
         [1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
         [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
         [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
         [0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
         [0, 0, 1, 0, 1, 1, 1, 0, 1, 1]]
    print(f"Solve for matrix by using Gaussian Elimination: ")
    x = find_x(A)
    print(f"The final vector we found is: {x}")


if __name__ == '__main__':
    main()