import numpy as np
from datetime import datetime

a = np.random.randn(100)
b = np.random.randn(100)
T = 100000

def slow_dot_product(a, b):
    result = 0
    for e, f in zip(a, b):
        result += e*f
    return result
# old school way
t0 = datetime.now()
for t in range(T):
    slow_dot_product(a, b)
dt1 = datetime.now() - t0


# use numpy do the dot operation
t0 = datetime.now()
for t in range(T):
    a.dot(b)
dt2 = datetime.now() - t0

print ("dt1 / dt2:", dt1.total_seconds() / dt2.total_seconds())
'''dt1 / dt2: 36.371554357592096'''
'''dt1 / dt2: 35.13985478226045'''
'''dt1 / dt2: 36.90819261924548'''