import numpy as np

arr = np.arange(1,10)

arr=arr.reshape(3,3)

np.sum(arr)
np.mean(arr)
np.max(arr)
np.min(arr)

np.sum(arr, axis=0)
np.sum(arr, axis=1)

arr2= np.array([1,2,3,4,5,6,7,8])

arr2.reshape(4,2)

arr3=np.array([[1,2],[3,4]])
arr4=np.array([[5,6],[7,8]])

np.dot(arr3,arr4)

arr5=np.arange(1,7).reshape(2,3)
arr5.T