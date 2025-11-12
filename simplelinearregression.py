import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2

np.random.seed(42)
X_np=np.random.rand(100, 1).astype(np.float32)
Y_np=3*X_np+2+0.1*np.random.randn(100,1).astype(np.float32)
X=torch.from_numpy(X_np)
Y=torch.from_numpy(Y_np)

model=nn.Linear(in_features=1,out_features=1)

criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)

for epoch in range(200):
    outputs=model(X)
    loss=criterion(outputs,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%50==0:
        print(f"Epoch[{epoch+1}/200],Loss:{loss.item():.4f}")

predicted=model(X).detach().numpy()
plt.figure(figsize=(6,4))
plt.scatter(X_np,Y_np,label='Original data')
plt.plot(X_np,predicted,color='red',label='Fitted line')
plt.legend()
plt.title("Simple Linear Regression with PyTorch")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


















