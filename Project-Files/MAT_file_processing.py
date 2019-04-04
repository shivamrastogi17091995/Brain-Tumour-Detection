import numpy as np
import h5py
X_list = []
Y_list = []
for num in range(1,3065):
    if (num >= 955 and num <= 957) or (num >= 1070 and num <= 1076) or (num >= 1203 and num <= 1207):
        continue
    file = h5py.File('MRI DATA/'+str(num)+'.mat','r')
    image = file['/cjdata/image/']
    X_list.append(image)
    label = file['/cjdata/label/']
    Y_list.extend(label[0])
print(len(X_list))
print(len(Y_list))
X_list = np.array(X_list)
print(X_list.shape," ",X_list.dtype)
np.save('MRI X_list.npy',X_list)
Y_list = np.array(Y_list)
print(Y_list.shape," ",Y_list.dtype)
np.save('MRI Y_list.npy',Y_list)
