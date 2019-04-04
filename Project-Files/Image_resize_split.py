import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
X_list = np.load('MRI X_list.npy')
Y_list = np.load('MRI Y_list.npy')
X_list_new = []
for img in X_list:
    temp = resize(img, (96, 96), anti_aliasing=True)
    X_list_new.append(temp)
X_list = np.array(X_list_new)
print(X_list.shape," ",X_list.dtype)
X_train,X_test,Y_train,Y_test = train_test_split(X_list,Y_list)
np.save('X_train.npy',X_train)
np.save('X_test.npy',X_test)
np.save('Y_train.npy',Y_train)
np.save('Y_test.npy',Y_test)
