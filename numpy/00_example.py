import numpy as np
from datetime import datetime




def playaround():
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


def main():
    playaround()

    return 0



main()



