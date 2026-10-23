import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import onnx
import json


def analytical_solution(x: np.ndarray,
                        y: np.ndarray,
                        t: np.ndarray,
                        bounds: dict,
                        alpha: float) -> np.ndarray:
    """
    Аналитическое решение

    Параметры:
        x: np.ndarray
            Вектор координат x.
        y: np.ndarray
            Вектор координат y.
        t: np.ndarray
            Вектор координат t.
        bounds: dict
            Границы расчетной области.
        alpha: float
            Коэффициент температурапроводности материала.

    Возвращает:
        np.ndarray
            Вектор значений температур, являющихся аналитическим решением.
    """
    u = (np.sin(np.pi * x / bounds['x']) * np.sin(np.pi * y / bounds['y'])
         * np.exp(-2 * np.pi ** 2 * alpha * t / (bounds['x'] * bounds['y'])))

    return u


class SinLayer(torch.nn.Module):
    """
    Слой нейронной сети с синусоидальной функцией активации.

    Параметры:
        in_features: int
            Количество входных признаков.
        out_features: int
            Количество выходных признаков.
        bias: bool
            Использовать ли смещение в линейном слое.
        is_first: bool
            Является ли слой первым в сети.
        omega_0: float
            Частота для функции синуса.
    """

    def __init__(self,
                 input: float,
                 output: float,
                 bias: bool = True,
                 is_first: bool = False,
                 omega_0: float = 30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = input
        self.linear = torch.nn.Linear(input, output, bias=bias)
        self.init_weights()

    def init_weights(self):
        """
        Инициализация весов линейного слоя.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features)
                                            / self.omega_0,
                                            np.sqrt(6 / self.in_features)
                                            / self.omega_0)

    def forward(self, input):
        """
        Прямой проход через слой.

        Параметры:
            input: torch.Tensor
                Входной тензор.

        Возвращает:
            torch.Tensor
                Результат применения синусоидальной функции активации.
        """
        return torch.sin(self.omega_0 * self.linear(input))