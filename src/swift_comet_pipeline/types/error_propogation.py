from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValueAndStandardDev:
    value: float
    sigma: float

    def __add__(self, other):
        if not isinstance(other, ValueAndStandardDev):
            return ValueAndStandardDev(value=self.value + other, sigma=self.sigma)

        value = self.value + other.value
        sigma = np.sqrt(self.sigma**2 + other.sigma**2)
        return ValueAndStandardDev(value=value, sigma=sigma)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # self - other
        if not isinstance(other, ValueAndStandardDev):
            return ValueAndStandardDev(value=self.value - other, sigma=self.sigma)

        return ValueAndStandardDev(
            value=self.value - other.value,
            sigma=np.sqrt(self.sigma**2 + other.sigma**2),
        )

    def __rsub__(self, other):
        # other - self
        return ValueAndStandardDev(value=other - self.value, sigma=self.sigma)

    def _relative_error(self):
        return self.sigma / self.value

    def __mul__(self, other):
        if not isinstance(other, ValueAndStandardDev):
            return ValueAndStandardDev(
                value=self.value * other, sigma=self.sigma * other
            )

        v = self.value * other.value
        s = np.abs(v) * np.sqrt(
            self._relative_error() ** 2 + other._relative_error() ** 2
        )
        return ValueAndStandardDev(value=v, sigma=s)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # self/other
        if not isinstance(other, ValueAndStandardDev):
            return ValueAndStandardDev(
                value=self.value / other, sigma=self.sigma / other
            )

        v = self.value / other.value
        s = np.abs(v) * np.sqrt(
            self._relative_error() ** 2 + other._relative_error() ** 2
        )
        return ValueAndStandardDev(value=v, sigma=s)

    def __rtruediv__(self, other):
        # other/self
        return other.__truediv__(self)
