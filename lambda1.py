import os


def clear():
    return os.system("cls")


clear()

# def remainder(num):
#    return num % 2


def remainder(num):
    return num % 2


# remainder = lambda num: num % 2


print(type(remainder))
print(remainder(5))


def product(x, y):
    return x * y


# product = lambda x,y: x *y
print(product(2, 3))


def myfuction(num):
    return lambda x: x * num


result10 = myfuction(10)
result100 = myfuction(100)
# result10 = lambda x: x * 10
print(result10(9))
print(result100(9))


def myfunc(n):
    return lambda a: a * n


mydoubler = myfunc(2)  # this is n
mytripler = myfunc(3)  # this is n

print(mydoubler(11))  # this is a
print(mytripler(11))  # this is a

numbers = [2, 4, 6, 8, 10, 3, 18, 14, 21]

filtered_list = list(filter(lambda num: (num > 7), numbers))
print(filtered_list)

mapped_list = list(map(lambda num: num % 2, numbers))
print(mapped_list)

x = lambda a: a + 10
print(x(5))

x = lambda a, b, c: a + b + c
print(x(5, 6, 7))


def addition(n):
    return n + n


numbers = [1, 2, 3, 4]
result = map(addition, numbers)
# result = map(lambda x: x + x, numbers)
print(list(result))
