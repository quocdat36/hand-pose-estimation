# FILE: src/one_euro_filter.py

import math

class LowPassFilter:
    def __init__(self, alpha):
        self.__set_alpha(alpha)
        self.__y = None
        self.__s = None

    def __set_alpha(self, alpha):
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]" % alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.__set_alpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        self.__s = s
        return s

    def last_value(self):
        return self.__y


class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        if freq <= 0:
            raise ValueError("freq should be > 0")
        if min_cutoff <= 0:
            raise ValueError("min_cutoff should be > 0")
        if d_cutoff <= 0:
            raise ValueError("d_cutoff should be > 0")
        self.__freq = float(freq)
        self.__min_cutoff = float(min_cutoff)
        self.__beta = float(beta)
        self.__d_cutoff = float(d_cutoff)
        self.__x = LowPassFilter(self.__alpha(self.__min_cutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__d_cutoff))
        self.__last_time = None

    def __alpha(self, cutoff):
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        if self.__last_time is not None and timestamp is not None:
            self.__freq = 1.0 / (timestamp - self.last_time)
        self.__last_time = timestamp
        prev_x = self.__x.last_value()
        dx = 0.0 if prev_x is None else (x - prev_x) * self.__freq
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__d_cutoff))
        cutoff = self.__min_cutoff + self.__beta * abs(edx)
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))