import numpy as np

# Define the low-resolution images L1, L2, L3, and L4
l1 = np.array([[1, 3],
               [2, 4]])

l2 = np.array([[5, 7],
               [6, 8]])

l3 = np.array([[300, 301],
               [310, 311]])

l4 = np.array([[400, 401],
               [410, 411]])

# Create an empty higher resolution image Hˆ
H_hat = np.zeros((4, 4), dtype=np.int32)

'''x = 0
for i in range(4):
    y = 0
    for j in range(0,4,2):
        if i % 2 == 0: #par
            H_hat[i,j] = l1[x,y]
            H_hat[i,j+1] = l3[x,y]
            y+=1
        else: #impar
            H_hat[i,j] = l2[x,y]
            H_hat[i,j+1] = l4[x,y]'''

# Compose the higher resolution image Hˆ using l1 only for even rows
'''for i in range(0, 4, 2):  # Iterate over even rows
    for j in range(2,4,2):
        H_hat[i,j-2] = l1
        H_hat[i:i+2, 0:2] = l1'''
H_hat[::2, ::2] = l1
H_hat[1::2, ::2] = l2
H_hat[::2, 1::2] = l3
H_hat[1::2, 1::2] = l4

# Compose the higher resolution image Hˆ using l2 only for odd rows
'''for i in range(1, 4, 2):  # Iterate over odd rows
    H_hat[i:i+2, 0:2] = l2'''

# Print the composed higher resolution image Hˆ
#print(H_hat)
x=0
for i in range(l1.shape[0]):
    for j in range(l1.shape[0]):
        x+=(l1[i,j]-l2[i,j])**2
x=(x/l1.shape[0]**2)**0.5

print(x)
