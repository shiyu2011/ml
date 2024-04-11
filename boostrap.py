import numpy as np

#original dataset
data = np.array([1,2,3,4,5])

#generate a bootstrap sample
bootstrap_sample = np.random.choice(data, size=data.shape[0], replace=True)

print(data)
print(bootstrap_sample)
