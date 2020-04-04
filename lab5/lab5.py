import numpy as np
from tabulate import tabulate

a = [[0, 0.17, -0.33, 0.18],
	 [0, 0, 0.5244, -0.0976],
	 [0.2785, 0.2278, 0, 0.0886],
	 [0.0833, 0.0729, 0.21875, 0]]

b = [-1.2, 0.4024, 0.6075, -1.25]
# x0 = b

l_0 = 0
l_2 = 0
l_inf = 0
maximum = 0
s = 0

print()
for i in range(len(a)):
	for j in range(len(a)):
		l_2 += a[i][j]**2
print("l_2 = ", l_2)

for i in range(len(a)):
	for j in range(len(a)):
		s += abs(a[i][j])
	if s > maximum:
		maximum = s
	s = 0
l_inf = maximum
print("l_inf = ", l_inf)

maximum = 0
for i in range(len(a)):
	for j in range(len(a)):
		s += abs(a[j][i])
	if s > maximum:
		maximum = s
	s = 0
l_0 = maximum
print("l_0 = ", l_0)
print()

if l_0 < 1 or l_2 < 1 or l_inf < 1:
	print("At least one of norm is less than 1 --> System has unique solution")

aa = np.array([[-1, 0.17, -0.33, 0.18],
			   [0, -0.82, 0.43, -0.08],
			   [0.22, 0.18, -0.79, 0.07],
			   [0.08, 0.07, 0.21, -0.96]])

bb = np.array([1.2,
			   -0.33,
			   -0.48,
			   1.2])
# print(np.linalg.solve(aa, bb))
# print()

sum = 0
for i in range(len(aa)):
	for j in range(len(aa)):
		sum += (aa[i][j])**2

aa_inv = np.array([])
aa_inv = np.linalg.inv(aa)
sum_inv = 0
for i in range(len(aa_inv)):
	for j in range(len(aa_inv)):
		sum_inv += aa_inv[i][j]**2

print("The estimated number of iterations : ", int((sum*sum_inv)**0.5) + 1)
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
