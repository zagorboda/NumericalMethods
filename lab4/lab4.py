from tabulate import tabulate
import numpy as np
# a = [[1,-2.01,2.04,0.17,0.18],
# 	 [0.33,-0.77,0.44,-0.51,0.19],
# 	 [0.31,0.17,-0.21,0.54,0.21],
# 	 [0.17,1,-0.13,0.21,0.31]]

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

print("------- A matrix -------")
print(tabulate(matrix))
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
print(tabulate(l))
print()

print("------- U matrix -------")
print(tabulate(u))
print()

u = np.array(u)
l = np.array(l)
result = np.dot(l,u)
print("------- L x U -------")
print(tabulate(result))
print()

print("-- Determinate using LU decomposition --")
d = 1
for i in range(len(u)):
	d *= u[i][i]
d = round(d, 7)
print("D = ", d)
print()

print("-- Solution using LU decomposition --")
u = np.array(u)
l = np.array(l)
b = np.array(b)
y = np.array(np.linalg.solve(l, b))
x = np.linalg.solve(u, y)
print(tabulate(x))

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
print(tabulate(x))
print("--- Inverse matrix using Numpy function ---")
print(tabulate(np.linalg.inv(matrix)))


# R = 8
for R in range(3,8):
	print(" ------- Matrix {}*{}  ------- ".format(R, R))
	arr = []
	c = []
	for i in range(R):
		for j in range(R):
			c.append(11.7 / (1 + 2.3*i*j)**7)
		arr.append(c[:])
		c.clear()
	# print(tabulate(arr))
	sum = 0
	for i in range(len(arr)):
		for j in range(len(arr)):
			sum += arr[i][j]

	arr_inv = np.array([])
	arr_inv = np.linalg.inv(arr)
	# print(tabulate(arr_inv))
	sum_inv = 0
	for i in range(len(arr_inv)):
		for j in range(len(arr_inv)):
			sum_inv += arr_inv[i][j]

	print("--- Euclidean norm ---")
	print(sum**0.5)
	print("--- Euclidean norm inverse ---")
	print(sum_inv**0.5)

	print("- Number of conditionality of the matrix -")
	print((sum**0.5)*(sum_inv**0.5))
	if (sum**0.5)*(sum_inv**0.5) < 100:
		print("The matrix is WELL conditioned")
	else:
		print("The matrix is BADLY conditioned")
