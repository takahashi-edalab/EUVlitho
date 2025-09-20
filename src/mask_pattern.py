import numpy as np
from abc import ABC, abstractmethod
import random


class MaskPattern(ABC):
    """Abstract base class for mask pattern generation"""

    @abstractmethod
    def _generate(self, mask_w: int, mask_h: int) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def __call__(self, mask_w: int, mask_h: int) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")


class LinePattern(MaskPattern):

    def __init__(
        self,
        cd: int = 56,
        gap: int = 80,
        direction: str = "H",
        field_type: str = "BF",
    ):
        self.cd = cd
        self.gap = gap
        self.direction = direction
        self.field_type = field_type

    def rand_mask_line(self, n_pixels: int, gap: int) -> np.ndarray:
        mask1d = np.zeros(n_pixels, dtype=int)
        line = random.randint(0, 1)
        sum_val = 0
        while sum_val < n_pixels - gap:
            a = gap * random.randint(0, 4)
            if line == 1:
                a = a + 3 * gap

            for i in range(sum_val, min(sum_val + a, n_pixels)):
                if line == 1:
                    mask1d[i] = 1
                else:
                    mask1d[i] = 0

            line = 1 - line
            sum_val += a

        for i in range(sum_val, n_pixels):
            if line == 0:
                mask1d[i] = 1

        return mask1d

    def _generate(self, mask_w: int, mask_h: int) -> np.ndarray:
        mask2d = np.zeros((mask_h, mask_w), dtype=int)
        space = random.randint(0, 1)
        sum_val = 0

        while sum_val < mask_w - self.cd:
            if space == 1:
                space = 0
            else:
                mask1dy = self.rand_mask_line(mask_h, gap=self.gap)
                for i in range(sum_val, min(sum_val + self.cd, mask_w)):
                    for j in range(mask_h):
                        mask2d[j, i] = mask1dy[j]
                space = 1
            # update
            sum_val += self.cd

        for i in range(sum_val, mask_w):
            if space == 1:
                for j in range(mask_h):
                    mask2d[i, j] = mask1dy[j]

        return mask2d

    def __call__(self, mask_w: int, mask_h: int) -> np.ndarray:
        mask = self._generate(mask_w, mask_h)
        if self.direction == "H":
            mask = mask.T

        if self.field_type == "BF":
            mask = 1 - mask

        return mask


class RandomLineSpacePattern(MaskPattern):

    def __init__(
        self,
        cd: int = 60,
        fac: float = 1.0,
        fac1d: float = 3.0,
        field_type: str = "BF",
    ):
        self.cd = cd
        self.fac = fac
        self.fac1d = fac1d
        self.field_type = field_type

    def randmask(self, mask_edge_len: int) -> np.ndarray:
        mask1d = np.zeros(mask_edge_len, dtype=int)
        line = random.randint(0, 1)
        sum_val = 0

        while sum_val < mask_edge_len - self.cd:
            a = int(self.cd * np.exp(self.fac1d * random.random()))

            for i in range(sum_val, min(sum_val + a, mask_edge_len)):
                if line == 1:
                    mask1d[i] = 0
                else:
                    mask1d[i] = 1

            line = 1 - line
            sum_val += a

        for i in range(sum_val, mask_edge_len):
            if line == 0:
                mask1d[i] = 0
            else:
                mask1d[i] = 1
        return mask1d

    def _generate(self, mask_w: int, mask_h: int) -> np.ndarray:
        mask2d = np.zeros((mask_h, mask_w), dtype=int)
        mask1dx = np.zeros(mask_w, dtype=int)
        mask1dy = np.zeros(mask_h, dtype=int)

        # Y direction processing
        space = random.randint(0, 1)
        sum_val = 0

        while sum_val < mask_h - self.cd:
            a = int(self.cd * np.exp(self.fac * random.random()))
            if space == 1:
                for i in range(sum_val, min(sum_val + a, mask_h)):
                    for j in range(mask_w):
                        mask2d[j, i] = 1
                space = 0
            else:
                mask1dx = self.randmask(mask_w)
                for i in range(sum_val, min(sum_val + a, mask_h)):
                    for j in range(mask_w):
                        mask2d[j, i] = mask1dx[j]
                space = 1
            sum_val += a

        for i in range(sum_val, mask_h):
            if space == 0:
                for j in range(mask_w):
                    mask2d[j, i] = 1
            else:
                for j in range(mask_w):
                    mask2d[j, i] = mask1dx[j]

        # X direction processing
        space = random.randint(0, 1)
        sum_val = 0

        while sum_val < mask_w - self.cd:
            a = int(self.cd * np.exp(self.fac * random.random()))
            if space == 1:
                space = 0
            else:
                mask1dy = self.randmask(mask_h)
                for i in range(sum_val, min(sum_val + a, mask_w)):
                    for j in range(mask_h):
                        mask2d[i, j] = min(mask1dy[j], mask2d[i, j])
                space = 1
            sum_val += a

        for i in range(sum_val, mask_w):
            if space == 0:
                space = 1
            else:
                for j in range(mask_h):
                    mask2d[i, j] = min(mask1dy[j], mask2d[i, j])
        return mask2d

    def __call__(self, mask_w: int, mask_h: int) -> np.ndarray:
        mask = self._generate(mask_w, mask_h)
        if self.field_type == "DF":
            mask = 1 - mask

        return mask
