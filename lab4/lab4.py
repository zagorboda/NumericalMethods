import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

matrix = np.array([[1,-2.01,2.04,0.17],
	 [0.33,-0.77,0.44,-0.51],
	 [0.31,0.17,-0.21,0.54],
	 [0.17,1,-0.13,0.21]])

b = np.array([[0.18],
	 [0.19],
	 [0.21],
	 [0.31]])

l = [[1,0,0,0],
	 [0,1,0,0],
	 [0,0,1,0],
	 [0,0,0,1]]

print()
print("------- A matrix -------")
print(tabulate(matrix, tablefmt="fancy_grid"))
a = np.array(matrix)

n = 0
q = 0
for i in range(1, len(a)):
	for k in range(i, len(a)):
		try:
			c = a[k][q]/a[i-1][q]
		except ZeroDivisionError:
			continue
		l[k][q] = c
		for j in range(n, len(a[0])):
			a[k][j]  = a[k][j] - c*a[i-1][j]
			n += 1
		n=0
	q += 1

u = a
print()
print("------- L matrix -------")
print(tabulate(l, tablefmt="fancy_grid"))
print()

print("------- U matrix -------")
print(tabulate(u, tablefmt="fancy_grid"))
print()

u = np.array(u)
l = np.array(l)
result = np.dot(l,u)
print("------- L x U -------")
print(tabulate(result, tablefmt="fancy_grid"))
print()

print("-- Determinate using LU decomposition --")
d = 1
for i in range(len(u)):
	d *= u[i][i]
d = round(d, 7)
print(tabulate([["D = ", d]], tablefmt="fancy_grid"))
print()

print("-- Solution using LU decomposition --")
u = np.array(u)
l = np.array(l)
b = np.array(b)
y = np.array(np.linalg.solve(l, b))
x = np.linalg.solve(u, y)
print(tabulate(x, tablefmt="fancy_grid"))

e = []
t = []
for i in range(len(matrix)):
	for j in range(len(matrix)):
		if i == j :
			t.append(1)
		else:
			t.append(0)
t = np.array(t)
t = np.reshape(t, (len(matrix), len(matrix)))
for i in range(len(t)):
	e.append(t[:,i])
e = np.array(e)
y = []
for i in range(len(e)):
	y.append(np.linalg.solve(l, np.reshape(e[i], (len(matrix),1))))

y = np.array(y)
print()

x = []
for i in range(len(y)):
	x.append(np.linalg.solve(u, np.reshape(y[i], (len(matrix),1))))
x = np.array(x)
x.tolist()
print("--- Inverse matrix using LU ---")
x = [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]
print(tabulate(x, tablefmt="fancy_grid"))
print("--- Inverse matrix using Numpy function ---")
print(tabulate(np.linalg.inv(matrix), tablefmt="fancy_grid"))
print()

ecl = []
cnd = []
print("--- a[i][j] = 11.7 / (1 + 2.3*i*j)**7 ---")
for R in range(3,8):
	arr = []
	c = []
	for i in range(R):
		for j in range(R):
			c.append(11.7 / (1 + 2.3*i*j)**7)
		arr.append(c[:])
		c.clear()
	if R == 5:
		print(" --- Task ---")
		print(tabulate(arr, tablefmt="fancy_grid"))
	sum = 0
	for i in range(len(arr)):
		for j in range(len(arr)):
			sum += arr[i][j]

	arr_inv = np.array([])
	arr_inv = np.linalg.inv(arr)
	sum_inv = 0
	for i in range(len(arr_inv)):
		for j in range(len(arr_inv)):
			sum_inv += arr_inv[i][j]

	if (sum**0.5)*(sum_inv**0.5) < 100:
		cond = "WELL"
	else:
		cond = "BADLY"
	
	ecl.append(sum**0.5)
	cnd.append((sum**0.5)*(sum_inv**0.5))
	print(tabulate([[" Matrix {}*{} ".format(R, R)],
					[" Euclidean norm =  ", sum**0.5], 
					[" Number of conditionality of the matrix = ", (sum**0.5)*(sum_inv**0.5)],
					[" The matrix is {} conditioned".format(cond)]], tablefmt="fancy_grid"))
	print()

r = [3,4,5,6,7]
plt.plot(r, ecl,'-ok', color="red")
plt.plot(r, cnd,'-ok', color="blue")
plt.xlabel("Matrix dimension")
plt.ylabel("Value")
plt.legend(['Euclidean norm','Number of conditionality'], loc='upper left')
plt.show()

print()
print(" --- K1*Si*cos(f1) + K2*sin(f1) - K3 = (Si)**2 , i=1,2,3 ---")
eq = np.zeros((3, 4))
s = 0
f = 0
for i in range(3):
	if i == 0:
		s = 0.9
		f = 30
	if i == 1:
		s = 1.15
		f = 45
	if i == 2:
		s = 1.9
		f = 60

	eq[i][0] = s*np.cos(np.deg2rad(f))
	eq[i][1] = np.sin(np.deg2rad(f))
	eq[i][2] = -1
	eq[i][3] = s**2
print()
print(" -- Math model --")
print(tabulate(eq, tablefmt="fancy_grid"))
A = np.zeros((len(eq), len(eq)))
B = np.zeros((len(eq), 1))

for i in range(len(eq[0])-1):
	A[:,i] = eq[:,i]
B[:,0] = eq[:,len(eq[0])-1]
K = np.linalg.solve(A, B)

print(tabulate([["K1", K[0]],
				["K2", K[1]],
				["K3", K[2]]], tablefmt="fancy_grid"))
print()
print("a1 = K1/2   a2 = sqrt(a1**2 + a3**2 - K3)  a3 = K2/(2a1)")

a1 = K[0]/2
a3 = K[1]/(2*a1)
a2 = np.sqrt(a1**2 + a3**2 - K[2])
print(tabulate([["a1", a1],
				["a2", a2],
				["a3", a3]], tablefmt="fancy_grid"))