import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from math import pi


class DerivativeNetwork(nn.Module):   # нейросеть с производной
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)


def loss_with_derivative(model, x):  #Функция потерь с производной
    x = x.clone().detach().requires_grad_(True)
    u = model(x)

    # Вычисляем du/dx через autograd
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Хотим, чтобы u(x) = sin(x), а du/dx = cos(x)
    loss_u = ((u - torch.sin(x)) ** 2).mean()
    loss_deriv = ((du_dx - torch.cos(x)) ** 2).mean()

    return loss_u + loss_deriv

model = DerivativeNetwork()

# Обучаем аппроксимировать sin(x)
x_train = torch.linspace(-3, 3, 100).reshape(-1, 1)  # создает 100 равномерно распред значений от -3 до 3, shape = (100,), reshape меняет форму на (100,1) (100 строк, 1 столбец)
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