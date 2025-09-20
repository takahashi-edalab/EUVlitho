import numpy as np
from abc import ABC, abstractmethod
import random


class MaskPattern(ABC):
    """Abstract base class for mask pattern generation"""

    @abstractmethod
    def maskgen(self, ndivx: int, ndivy: int) -> np.ndarray:
        """Generate 2D mask pattern

        Args:
            ndivx: X dimension size
            ndivy: Y dimension size

        Returns:
            2D numpy array representing the mask
        """
        pass


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
            else:
                mask1d[i] = 0

        return mask1d

    def __generate(self, ndivx: int, ndivy: int) -> np.ndarray:
        mask2d = np.zeros((ndivy, ndivx), dtype=int)
        space = random.randint(0, 1)
        sum_val = 0

        while sum_val < ndivx - self.cd:
            if space == 1:
                space = 0
            else:
                mask1dy = self.rand_mask_line(ndivy, gap=self.gap)
                for i in range(sum_val, min(sum_val + self.cd, ndivx)):
                    for j in range(ndivy):
                        mask2d[j, i] = mask1dy[j]
                space = 1
            # update
            sum_val += self.cd

        for i in range(sum_val, ndivx):
            if space == 1:
                for j in range(ndivy):
                    mask2d[i, j] = mask1dy[j]

        return mask2d

    def maskgen(self, ndivx: int, ndivy: int) -> np.ndarray:
        mask = self.__generate(ndivx, ndivy)
        if self.direction == "H":
            mask = mask.T

        if self.field_type == "BF":
            mask = 1 - mask

        return mask


class LSDFPattern(MaskPattern):
    """Line Space Dark Field pattern"""

    def __init__(self, cd: int = 60, fac: float = 1.0, fac1d: float = 3.0):
        self.cd = cd
        self.fac = fac
        self.fac1d = fac1d

    def maskgen(self, ndivx: int, ndivy: int) -> np.ndarray:
        mask2d = np.zeros((ndivy, ndivx), dtype=int)
        mask1dx = np.zeros(ndivx, dtype=int)
        mask1dy = np.zeros(ndivy, dtype=int)

        # Y direction processing
        space = random.randint(0, 1)
        sum_val = 0

        while sum_val < ndivy - self.cd:
            a = int(self.cd * np.exp(self.fac * random.random()))
            if space == 1:
                for i in range(sum_val, min(sum_val + a, ndivy)):
                    for j in range(ndivx):
                        mask2d[i, j] = 1
                space = 0
            else:
                mask1dx = self.randmask(ndivx, cd=self.cd, fac1d=self.fac1d)
                for i in range(sum_val, min(sum_val + a, ndivy)):
                    for j in range(ndivx):
                        mask2d[i, j] = mask1dx[j]
                space = 1
            sum_val += a

        for i in range(sum_val, ndivy):
            if space == 0:
                for j in range(ndivx):
                    mask2d[i, j] = 1
            else:
                for j in range(ndivx):
                    mask2d[i, j] = mask1dx[j]

        # X direction processing
        space = random.randint(0, 1)
        sum_val = 0

        while sum_val < ndivx - self.cd:
            a = int(self.cd * np.exp(self.fac * random.random()))
            if space == 1:
                space = 0
            else:
                mask1dy = self.randmask(ndivy, cd=self.cd, fac1d=self.fac1d)
                for i in range(sum_val, min(sum_val + a, ndivx)):
                    for j in range(ndivy):
                        mask2d[j, i] = min(mask1dy[j], mask2d[j, i])
                space = 1
            sum_val += a

        for i in range(sum_val, ndivx):
            if space == 0:
                space = 1
            else:
                for j in range(ndivy):
                    mask2d[j, i] = min(mask1dy[j], mask2d[j, i])

        return mask2d

    def randmask(self, ndiv: int, **kwargs) -> np.ndarray:
        cd = kwargs.get("cd", self.cd)
        fac1d = kwargs.get("fac1d", self.fac1d)
        mask1d = np.zeros(ndiv, dtype=int)

        line = random.randint(0, 1)
        sum_val = 0

        while sum_val < ndiv - cd:
            a = int(cd * np.exp(fac1d * random.random()))

            for i in range(sum_val, min(sum_val + a, ndiv)):
                if line == 1:
                    mask1d[i] = 0
                else:
                    mask1d[i] = 1

            line = 1 - line
            sum_val += a

        for i in range(sum_val, ndiv):
            if line == 0:
                mask1d[i] = 0
            else:
                mask1d[i] = 1

        return mask1d


class LSBFPattern(MaskPattern):
    """Line Space Bright Field pattern"""

    def __init__(self, cd: int = 60, fac: float = 1.0, fac1d: float = 3.0):
        self.cd = cd
        self.fac = fac
        self.fac1d = fac1d

    def maskgen(self, ndivx: int, ndivy: int) -> np.ndarray:
        mask2d = np.zeros((ndivy, ndivx), dtype=int)
        mask1dx = np.zeros(ndivx, dtype=int)
        mask1dy = np.zeros(ndivy, dtype=int)

        # Y direction processing
        space = random.randint(0, 1)
        sum_val = 0

        while sum_val < ndivy - self.cd:
            a = int(self.cd * np.exp(self.fac * random.random()))
            if space == 1:
                for i in range(sum_val, min(sum_val + a, ndivy)):
                    for j in range(ndivx):
                        mask2d[i, j] = 0
                space = 0
            else:
                mask1dx = self.randmask(ndivx, cd=self.cd, fac1d=self.fac1d)
                for i in range(sum_val, min(sum_val + a, ndivy)):
                    for j in range(ndivx):
                        mask2d[i, j] = mask1dx[j]
                space = 1
            sum_val += a

        for i in range(sum_val, ndivy):
            if space == 0:
                for j in range(ndivx):
                    mask2d[i, j] = 0
            else:
                for j in range(ndivx):
                    mask2d[i, j] = mask1dx[j]

        # X direction processing
        space = random.randint(0, 1)
        sum_val = 0

        while sum_val < ndivx - self.cd:
            a = int(self.cd * np.exp(self.fac * random.random()))
            if space == 1:
                space = 0
            else:
                mask1dy = self.randmask(ndivy, cd=self.cd, fac1d=self.fac1d)
                for i in range(sum_val, min(sum_val + a, ndivx)):
                    for j in range(ndivy):
                        mask2d[j, i] = max(mask1dy[j], mask2d[j, i])
                space = 1
            sum_val += a

        for i in range(sum_val, ndivx):
            if space == 0:
                space = 1
            else:
                for j in range(ndivy):
                    mask2d[j, i] = max(mask1dy[j], mask2d[j, i])

        return mask2d

    def randmask(self, ndiv: int, **kwargs) -> np.ndarray:
        cd = kwargs.get("cd", self.cd)
        fac1d = kwargs.get("fac1d", self.fac1d)
        mask1d = np.zeros(ndiv, dtype=int)

        line = random.randint(0, 1)
        sum_val = 0

        while sum_val < ndiv - cd:
            a = int(cd * np.exp(fac1d * random.random()))

            for i in range(sum_val, min(sum_val + a, ndiv)):
                if line == 1:
                    mask1d[i] = 1
                else:
                    mask1d[i] = 0

            line = 1 - line
            sum_val += a

        for i in range(sum_val, ndiv):
            if line == 0:
                mask1d[i] = 1
            else:
                mask1d[i] = 0

        return mask1d


# Pattern factory function
# def create_mask_pattern(pattern_type: str, **kwargs) -> MaskPattern:
#     """Factory function to create mask pattern instances

#     Args:
#         pattern_type: Type of pattern ('HBF', 'VDF', 'HDF', 'VBF', 'LSDF', 'LSBF')
#         **kwargs: Pattern-specific parameters

#     Returns:
#         MaskPattern instance
#     """
#     patterns = {
#         "HBF": HBFPattern,
#         "VDF": VDFPattern,
#         "HDF": HDFPattern,
#         "VBF": VBFPattern,
#         "LSDF": LSDFPattern,
#         "LSBF": LSBFPattern,
#     }

#     if pattern_type not in patterns:
#         raise ValueError(f"Unknown pattern type: {pattern_type}")

#     return patterns[pattern_type](**kwargs)
