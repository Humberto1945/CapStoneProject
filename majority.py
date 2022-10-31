
# Online Python - IDE, Editor, Compiler, Interpreter
import numpy as np
a1 = np.array([[1, 2, 7], [3, 4, 5],[3, 2, 5]])
a2 = np.array([[1, 3, 5], [3, 7, 2],[3, 2, 7]])
a3 = np.array([[1, 2, 7], [2, 7, 5],[2, 1, 7]])
a1 = a1.flatten()
a2 = a2.flatten()
a3 = a3.flatten()
# Suppose a2 is the prediction from highest accuracy model. So, that result is our default.
changes=np.where(a1 == a3) # Only modify the elements of a2 where a1 and a3 are same
# Modify a2 elements
for i in range(0,changes[0].size):
            a2[changes[0][i]]=a1[changes[0][i]]
a2 = a2.reshape((3,3))
print(a2)
