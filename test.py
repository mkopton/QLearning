import numpy as np 

data = np.load('SARSA_Q_est.npy')

rows, cols = data.shape

print(rows)
print(cols)
print(data)

for row in range(rows):
    print(f'For state {row}, direction {np.argmax(data[row,:])}')