import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from math import pi

class SmallNetwork(nn.Module): # все нейросети строятся на основе nn.Module
    def __init__(self):  # конструктор класса, описывает слои, из которых будет состоять сеть
        super().__init__()
        self.layer1 = nn.Linear(1, 20)  # 1 вход (x) -> 20 нейронов
        self.activation = nn.Tanh()  # активация (для PINN лучше tanh)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 1)  # 1 выход (y)

    def forward(self, x):   # модуль прямого прохода по нейросети задает последовательность работы слоев
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x


model = SmallNetwork()

# Обучаем аппроксимировать sin(x)
x_train = torch.linspace(-3, 3, 100).reshape(-1, 1)  # (100,1)
y_true = torch.sin(x_train)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    # forward
    y_pred = model(x_train)
    loss = ((y_pred - y_true) ** 2).mean()

    # backward
    optimizer.zero_grad()  # обнуляем старые градиенты
    loss.backward()  # вычисляем новые
    optimizer.step()  # обновляем веса

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss={loss.item():.4f}")

x_test = torch.linspace(-3, 3, 200).reshape(-1, 1)
y_test = model(x_test)

plt.plot(x_test.numpy(), y_test.detach().numpy(), label='Predicted')
plt.plot(x_train.numpy(), y_true.numpy(), 'r.', label='True')
plt.legend()
plt.show()