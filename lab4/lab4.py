from tabulate import tabulate
import numpy as np

# a = [[1,-2.01,2.04,0.17,0.18],
# 	 [0.33,-0.77,0.44,-0.51,0.19],
# 	 [0.31,0.17,-0.21,0.54,0.21],
# 	 [0.17,1,-0.13,0.21,0.31]]

a = [[1,-2.01,2.04,0.17],
	 [0.33,-0.77,0.44,-0.51],
	 [0.31,0.17,-0.21,0.54],
	 [0.17,1,-0.13,0.21]]

b = [[0.18],
	 [0.19],
	 [0.21],
	 [0.31]]

solution = np.linalg.solve(a, b)

l = [[1,0,0,0],
	 [0,1,0,0],
	 [0,0,1,0],
	 [0,0,0,1]]

print("------- A matrix -------")
print(tabulate(a))


n = 0
e = 0
for i in range(1, len(a)):
	for k in range(i, len(a)):
		try:
			c = a[k][e]/a[i-1][e]
		except ZeroDivisionError:
			continue
		l[k][e] = c
		for j in range(n, len(a[0])):
			a[k][j]  = a[k][j] - c*a[i-1][j]
			n += 1
		n=0
	e += 1

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

print("-- Solution using LU decomposition --")
u = np.array(u)
l = np.array(l)
b = np.array(b)
y = np.array(np.linalg.solve(l, b))
x = np.linalg.solve(u, y)
print(x)


