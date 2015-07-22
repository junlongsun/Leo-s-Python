from numpy import dot, array, matrix, multiply



# matrix multipy
A = array([[1, 2], [3,4]])
B = array([[3,4],[5,6]])

C = dot(A,B)
print C

D = matrix('1 2; 3 4')
E = matrix('3,4; 5,6')

F = dot(D,E)
print F

num = F.size
print num
