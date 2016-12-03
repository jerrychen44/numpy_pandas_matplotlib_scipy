import numpy as np
from datetime import datetime




def list_array_as_vector():
    L=[1,2,3]
    A=np.array([1,2,3])

    print(L)
    print(A)


    L.append(4)
    print(L)

    L=L + [5]
    print(L)



    print(A+A)

    print(A*5)

    print(L+L)



    print(np.sqrt(A))

    print(np.exp(A))

    print(np.log(A))


    return 0

def dot_product():
    ############################
    # dot = 點(內)積 = 矩陣乘法
    # [Dot product in 代數空間的算法]
    # a*b = aTb = sigma form d=1 to D ad bd
    # ref: https://zh.wikipedia.org/wiki/%E6%95%B0%E9%87%8F%E7%A7%AF
    #############################
    a=np.array([1,2])
    b=np.array([2,1])

    #0
    dot=0
    for e,f in zip(a,b):
        dot+=e*f

    print(dot)

    print(a,type(a))
    '''[1 2] <class 'numpy.ndarray'>'''
    print(b)

    #1
    print(a*b)
    print(np.sum(a*b))
    # or sum by it self
    print((a*b).sum())

    #2
    print(np.dot(a,b))

    #3
    print(a.dot(b))
    print(b.dot(a))



    print("------------")
    ###################
    # [Dot product in 幾何空間的算法]
    # a dot b =|a||b|cos(theta)ab
    # cos(theta)ab = a dot b / |a||b|
    # ref: https://zh.wikipedia.org/wiki/%E6%95%B0%E9%87%8F%E7%A7%AF, 中間搜尋:幾何定義
    ####################
    #0
    amag = np.sqrt( (a*a).sum())
    print((a*a).sum())
    print(amag)
    #1
    amag = np.linalg.norm(a)
    print(amag)


    cosangle = a.dot(b) / ( np.linalg.norm(a) * np.linalg.norm(b))
    print(cosangle)
    angle = np.arccos(cosangle)
    print(angle)
    return 0

def matrix_listoflist():

    M = np.array([[1,2],[3,4]])
    print(M,type(M))
    '''
    [[1 2]
     [3 4]] <class 'numpy.ndarray'>
     '''
    L = [[1,2],[3,4]]
    print(L[0][1])
    print(M[0][1])
    print(M[0,1])

    # use np.matrix looks like np.array, but different type
    M2 = np.matrix([[1,2],[3,4]])
    print(M2,type(M2))
    '''
    [[1 2]
     [3 4]] <class 'numpy.matrixlib.defmatrix.matrix'>
     '''
    # matrix to array
    M2array=np.array(M2)
    print(M2array,type(M2array))

    '''
    [[1 2]
     [3 4]] <class 'numpy.ndarray'>
     '''
     # Matrix is 2 dimational ndarry
     # vector is 1 dim ndarry

    return 0


def matrix_array_operation():
    #######################
    # initail matrix
    ########################
    # hand made array
    Z=np.array([1,2,3])
    print(Z)

    # initial array by zero
    Z = np.zeros(10)
    print(Z)

    # inital 2 dim array by zero
    Z = np.zeros((10,10))
    print(Z)

    # inital 2 dim array by one
    Z = np.ones((10,10))
    print(Z)


    # initail by random number
    R = np.random.random((10,10))
    print(R.mean(),R.var(),R.std())# default the random number is come from uniform distrubtion.array

    G = np.random.randn(10,10)
    print((G))# with Gussion distrubtion with mean = 0, varance =1, std=1
    print(G.mean(),G.var(),G.std())


    ################
    # Matrix Products
    ################
    # * => element wise multiplication
    # dot => math matrix multiplication

    # Get inverse
    A=np.array([[1,2],[3,4]])
    print(A)
    Ainv= np.linalg.inv(A)
    print(Ainv)
    # check it, AdotAt=I
    print(Ainv.dot(A))
    print(A.dot(Ainv))


    # Matrix determinate
    print(np.linalg.det(A))

    # Get diagonal
    print(np.diag(A))

    # create a diagonal matrix
    print(np.diag([1,2]))



    ###################
    # Do the outer product / inner product
    ###################
    a = np.array([1,2])
    b = np.array([3,4])
    print(np.outer(a,b))

    #inner
    print(np.inner(a,b))
    # the same with dot
    print(a.dot(b))


    #############
    # get the matrix trace = sum of diag
    #############
    print(np.diag(A).sum())
    print(np.trace(A))


    #################
    # Get covarence, need transpose the X
    # note: cov matrix is a Symmetric matrix
    #################
    # assume we have 100 data with 3 features,
    # and want to see the covarence of these 3 features.
    # the cov matrix should be 3x3
    X = np.random.randn(100,3)
    cov = np.cov(X)
    print(cov.shape)
    '''100,100 is wrong'''

    cov = np.cov(X.T)
    print(cov,cov.shape)
    '''
    [[ 0.98374545 -0.03866523  0.06900064]
     [-0.03866523  0.76934592  0.00223949]
     [ 0.06900064  0.00223949  1.06563292]] (3, 3)
     '''

    #######################
    # Eigenvalues and Eigenvectors
    # np.linalg.eig(C) for non-symmetrix array
    # np.linalg.eigh(C) for symmetrix and hermitian matrix
    #######################
    print(np.linalg.eigh(cov))

    '''
    First tuple is eigenvalues
    Second tuple is eigenvectors
    (array([ 0.98043336,  1.00387401,  1.37975078]),
    array([[-0.69095016,  0.55098584, -0.46797701],
           [-0.39734136,  0.25133919,  0.88258057],
           [-0.60391035, -0.79576581, -0.0452666 ]]))
    '''

    # how about we use regular eig()?
    # they are the same but with different sign and order.
    print(np.linalg.eig(cov))
    '''
    (array([ 1.37975078,  0.98043336,  1.00387401]),
    array([[-0.46797701,  0.69095016, -0.55098584],
           [ 0.88258057,  0.39734136, -0.25133919],
           [-0.0452666 ,  0.60391035,  0.79576581]]))
    '''
    return 0

def solve_linear_system():
    ######################
    # Ax=b
    # assume A is a invertible matrix DxD matrix
    # solution = Ainv Ax = x = Ainv b
    ####################
    # so we need matrix inverse  and multuply (dot)

    # hand made
    A=np.array([[1,2],[3,4]])
    b = np.array([1,2])
    print(A)
    print(b)
    x = np.linalg.inv(A).dot(b)
    print(x)

    # use numpy funciton do the same thing
    x = np.linalg.solve(A,b)
    print(x)
    # always use solve, not use inv(A) by yourself.
    # above case is just for demo.
    # use solve() is more effecient and more accurate



    return 0


def solving_linear_equation():
    '''
    The admission fee at a small fair is $1.50 for children and $4.00 for adults.
    On a certain day, 2200 people enter the fair and $5050 is collected.
    Q: How many children and how many aduts attended?

    Let:
    X1=number of children
    X2=number of adults
    X1+X2 =2200
    1.5 x X1+ 4 x X2 =5050


    This is a linear equation :
    | 1   1| | X1 | = | 2200 |
    | 1.5 4| | X2 |   | 5050 |
    '''
    A = np.array([[1,1],[1.5,4]])
    b = np.array([2200,5050])
    print(A)
    print(b)
    '''
        [[ 1.   1. ]
         [ 1.5  4. ]]
        [2200 5050]
    '''
    x = np.linalg.solve(A,b)
    print(x)
    '''
    X1=1500
    X2=700
    [ 1500.   700.]
    '''
    return 0
def main():
    #list_array_as_vector()
    #dot_product()
    #matrix_listoflist()
    #matrix_array_operation()
    #solve_linear_system()
    solving_linear_equation()
    return 0



main()



