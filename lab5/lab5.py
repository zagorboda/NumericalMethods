import numpy as np
from tabulate import tabulate

print()
print(" --- Iteration method --- ")

aa = np.array([[-1, 0.17, -0.33, 0.18],
			   [0, -0.82, 0.43, -0.08],
			   [0.22, 0.18, -0.79, 0.07],
			   [0.08, 0.07, 0.21, -0.96]])

bb = np.array([1.2,
			   -0.33,
			   -0.48,
			   1.2])
bb = np.reshape(bb, (4, 1))

print("A matrix")
print(tabulate(aa, tablefmt="fancy_grid"))
print("B matrix")
print(tabulate(bb, tablefmt="fancy_grid"))

a = [[0, 0.17, -0.33, 0.18],
	 [0, 0, 0.5244, -0.0976],
	 [0.2785, 0.2278, 0, 0.0886],
	 [0.0833, 0.0729, 0.21875, 0]]

b = [-1.2, 0.4024, 0.6075, -1.25]

l_0 = 0
l_2 = 0
l_inf = 0
maximum = 0
s = 0

print()
for i in range(len(a)):
	for j in range(len(a)):
		l_2 += a[i][j]**2

for i in range(len(a)):
	for j in range(len(a)):
		s += abs(a[i][j])
	if s > maximum:
		maximum = s
	s = 0
l_inf = maximum

maximum = 0
for i in range(len(a)):
	for j in range(len(a)):
		s += abs(a[j][i])
	if s > maximum:
		maximum = s
	s = 0
l_0 = maximum

print(tabulate([[" l_0 = ", l_0],
				[" l_2 = ", l_2],
				[" l_inf = ", l_inf]], tablefmt="fancy_grid"))

if l_0 < 1 or l_2 < 1 or l_inf < 1:
	print("At least one of norm is less than 1 --> System has unique solution / The method is converge")
print()

x1 = b[0]
x2 = b[1]
x3 = b[2]
x4 = b[3]

temp1 = x1
temp2 = x2
temp3 = x3
temp4 = x4

count = 0

while True:
	x1 = b[0] + (a[0][0]*x1 + a[0][1]*x2 + a[0][2]*x3 + a[0][3]*x4)
	x2 = b[1] + (a[1][0]*x1 + a[1][1]*x2 + a[1][2]*x3 + a[1][3]*x4)
	x3 = b[2] + (a[2][0]*x1 + a[2][1]*x2 + a[2][2]*x3 + a[2][3]*x4)
	x4 = b[3] + (a[3][0]*x1 + a[3][1]*x2 + a[3][2]*x3 + a[3][3]*x4)
	count += 1

	delta = max(abs(temp1-x1), abs(temp2-x2), abs(temp3-x3), abs(temp4-x4))
	if delta <= 0.001*(1 - l_inf) / l_inf:
		break

	temp1 = x1
	temp2 = x2
	temp3 = x3
	temp4 = x4

print("Number of iterations : ", count)
print(tabulate([[" x1 = ", x1],
				[" x2 = ", x2], 
				[" x3 = ", x3],
				[" x4 = ", x4]], tablefmt="fancy_grid"))
print("\n"*3)


a = np.array([[11.65, -1.76, 0.77],
	 		  [-1.76, 11.04, -2.61],
	 		  [0.77, -2.61, 13.18]])

b = np.array([10.66,
	 		 6.67,
			 11.34])
b = np.reshape(b, (3, 1))

print(" --- Matrix for Jacobi and  Gauss-Seidel methods --- ")
print("A matrix")
print(tabulate(a, tablefmt="fancy_grid"))
print("B matrix")
print(tabulate(b, tablefmt="fancy_grid"))
print()

print(" --- Jacobi method --- ")
print()

x = np.array([0,
			  0,
			  0])
x = np.reshape(x, (3, 1))

sum = 0
converge = True
for i in range(len(a)):
	for j in range(len(a)):
		if i!=j:
			sum += a[i][j]
	if a[i][i] < sum:
		converge = False
		break
	sum = 0

if converge:
	print("Method is converge for this matrix")
print()

d = np.diag(np.diag(np.copy(a)))
r = a - d
d_inv = np.linalg.inv(d)

count = 0
oversight = 0
while True:
	x_prev = x
	x = np.dot(d_inv, b - np.dot(r, x_prev))
	if count == 2:
		oversight = max(abs(x_prev - x))
	count += 1
	if max(abs(x_prev - x)) < 0.001:
		break
print("Oversight on 3 iteration : ", oversight)
print("Number of iterations : ", count)
print(tabulate([[" x1 = ", x[0]],
				[" x2 = ", x[1]], 
				[" x3 = ", x[2]]], tablefmt="fancy_grid"))
print()
residual = np.dot(a, x) - b
print("Residual : ")
print(tabulate(residual, tablefmt="fancy_grid"))


print(" --- Gauss-Seidel method --- ")
print()

l = np.copy(a)
u = np.copy(a)

for i in range(len(a)):
	for j in range(len(a)):
		if j > i:
			l[i][j] = 0
		if j <= i:
			u[i][j] = 0
l_inv = np.linalg.inv(l)

x = np.array([0,0,0])
x = np.reshape(x, (3,1))
x_prev = x

sum = 0
converge = True
for i in range(len(a)):
	for j in range(len(a)):
		if i!=j:
			sum += a[i][j]
	if a[i][i] < sum:
		converge = False
		break
	sum = 0

if converge:
	print("Method is converge for this matrix")
print()

count = 0
while True:
	x = np.dot(l_inv, (b - np.dot(u, x_prev)))
	if np.linalg.norm(x - x_prev) <= 0.001:
		break
	if count == 2:
		oversight = max(abs(x_prev - x))
	count += 1
	x_prev = x

print("Oversight on 3 iteration : ", oversight)
print("Number of iterations ", count)
print(tabulate([[" x1 = ", x[0]],
				[" x2 = ", x[1]], 
				[" x3 = ", x[2]]], tablefmt="fancy_grid"))
print()
residual = np.dot(a, x) - b
print("Residual : ")
print(tabulate(residual, tablefmt="fancy_grid"))
