"""
Author: Missy Shi

Course: math 458

Date: 04/23/2020

Project: A3 - 3

Description:
    Implementation of the Extended Euclidean Algorithm.

Task:
    1. Find the inverse of 197189 (mod 999979),
    2. Compute the numbers gcd(21000 − 229, 1000! + 98)
    3. Find gcd(887519, 146744, 388025, 880464, 189074)

Notes:
    use the tool: list reduce
"""


import math


def gcd(a: int, b: int):
    """"""
    if a == 0 or b == 0:
        return max(a, b)

    if b > a:
        temp = a
        a = b
        b = temp

    q = []
    r = []
    while b != 0:
        temp = b
        q.append(a // b)
        b = a % b
        r.append(b)
        a = temp
    return a, q, r


def gcdEx(quotient):
    listP = [0, 1]
    p0 = 0
    p1 = 1
    listQ = [1, 0]
    q0 = 1
    q1 = 0

    for q in quotient:
        curP = q * p1 + p0
        listP.append(curP)
        p0 = p1
        p1 = curP
        curQ = q * q1 + q0
        q0 = q1
        q1 = curQ
        listQ.append(curQ)
    # print(listP)
    # print(listQ)

    u = listQ[-2]
    v = listP[-2]
    return u, v


def a_bigger(a, b):
    if b > a:
        temp = a
        a = b
        b = temp
    return a, b


def task1():
    """Find the inverse of 197189 (mod 999979)"""
    a = 197189
    b = 999979
    a, b = a_bigger(a, b)
    g, q, r = gcd(a, b)
    u, v = gcdEx(q)
    print(f"the inverse of 197189 (mod 999979) is {v}")


    return


def task2():
    """Compute the numbers gcd(21000 − 229, 1000! + 98)"""
    a = (21000 - 229)  # modulus
    b = math.factorial(1000) + 98
    a, b = a_bigger(a, b)
    g, quotient, remainder = gcd(a, b)
    # print(f"gcd({a}, {b}) = {g}")
    print(f"gcd(21000 - 229, 1000! + 98) = {g}")
    return


def task3():
    """Find gcd(887519, 146744, 388025, 880464, 189074)"""
    num = [887519, 146744, 388025, 880464, 189074]
    # num = [21, 30, 24, 18]
    gcdlist = []
    num1 = num[0]
    num2 = num[1]
    g, q, r = gcd(num1, num2)
    for i in range(2, len(num)):
        # print(num[i])
        g, q, r = gcd(g, num[i])
        # gcdlist.append(g)
    # print(gcdlist)
    print(f"gcd(887519, 146744, 388025, 880464, 189074) = {g}")
    return


def q3():
    # task1()
    # task2()
    task3()

    return


def main():
    """ main program runner """
    q3()


if __name__ == '__main__':
    main()

