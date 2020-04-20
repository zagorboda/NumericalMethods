import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

a = np.array([[1, 2, -1],
              [-2, 4, -6],
              [-1, 0, -2]])

a1 = a
a1 = a1
p1 = a1[0][0] + a1[1][1] + a1[2][2]
temp = np.zeros((3, 3))
temp[0][0] = temp[1][1] = temp[2][2] = p1
b1 = a1-temp

a2 = np.dot(a, b1)
p2 = (a2[0][0] + a2[1][1] + a2[2][2])/2
temp = np.zeros((3, 3))
temp[0][0] = temp[1][1] = temp[2][2] = p2
b2 = a2-temp

a3 = np.dot(a, b2)
p3 = (a3[0][0] + a3[1][1] + a3[2][2])/3

print("Polynomial : P(a) = L^3 - 3L^2 - 3L + 8")

x = []
y = []
i = -3
while i < 5:
    x.append(i)
    y.append(i**3 - 3*i**2 - 3*i + 8)
    i += 0.01
# plt.plot(x, y, color="red")
# plt.grid(True, which='both')
# plt.axhline(y=0, color='k')
# plt.axvline(x=0, color='k')
# plt.show()

def func(c):
    return c**3 - 3*c**2 - 3*c + 8 

def first_derivative(c):
    return 3*c**2 - 6*c - 3

a = -3
b = 0
x0 = 0
if(func(a) > 0):
    x0 = b
elif(func(b) > 0):
    x0 = a
while True:
    x1 = x0 - func(x0)/first_derivative(x0)
    if abs(x1 - x0) < 2*10**(-3):
        print("x1 = %.3f" % x1)
        break
    x0 = x1

a = 0
b = 2
x0 = 0
if(func(a) > 0):
    x0 = b
elif(func(b) > 0):
    x0 = a
while True:
    x1 = x0 - func(x0)/first_derivative(x0)
    if abs(x1 - x0) < 2*10**(-3):
        print("x2 = %.3f" % x1)
        break
    x0 = x1
a = 2
b = 5

x0 = 0
if(func(a) > 0):
    x0 = a
elif(func(b) > 0):
    x0 = b
while True:
    x1 = x0 - func(x0)/first_derivative(x0)
    if abs(x1 - x0) < 2*10**(-3):
        print("x3 = %.3f" % x1)
        break
    x0 = x1

a = np.array([[1.6, 1.6, 1.7, 1.8],
              [1.6, 2.6, 1.3, 1.3],
              [1.7, 1.3, 3.6, 1.4],
              [1.8, 1.3, 1.4, 4.6]])

sum = 0
for i in range(len(a)):
		for j in range(len(a)):
			sum += a[i][j]**2

a_inv = np.array([])
a_inv = np.linalg.inv(a)
sum_inv = 0
for i in range(len(a_inv)):
    for j in range(len(a_inv)):
        sum_inv += a_inv[i][j]**2
print("Euclidean norm =  ", sum**0.5)
print("Number of conditionality of the matrix = ", (sum**0.5)*(sum_inv**0.5))
print()

values = np.linalg.eig(a)
print(values)


x0 = np.array([[1],
               [1],
               [1],
               [1]])

x1 = np.dot(a, x0)
y0 = np.linalg.norm(x1, 2)/np.linalg.norm(x0, 2)
e1 = 0.01
e2 = 0.001
while True:
    x2 = np.dot(a, x1)
    y1 = np.linalg.norm(x2, 2)/np.linalg.norm(x1, 2)
    if abs((y1-y0)/y1) < e1:
        y1_1 = y1
        x1_1 = x1
    if abs((y1-y0)/y1) < e2:
        break
    x1 = x2
    y0 = y1
print("e = 0.01")
print(y1_1)
print(tabulate(x1_1 / np.linalg.norm(x1_1, np.inf)))
print()

print("e = 0.001")
print(y1)
print(tabulate(x1 / np.linalg.norm(x1, np.inf)))

