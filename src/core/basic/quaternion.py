import numpy as np
from typing import Tuple, Union

from .linalg import (
    euler_to_quaternion,
    quaternion_to_euler,
    rotvec_to_quaternion,
    quaternion_to_rotvec,
    quaternion_to_dir_cos_mat,
    dir_cos_mat_to_quaternion,
    quarternion_conjugate,
    quaternion_multiply,
    quaternion_mul_vec,
)


class Quaternion:
    """
    四元数类，内部用 ndarray 存储为 (q0, q1, q2, q3)。
    """

    def __init__(self, q: Union[np.ndarray, Tuple[float, float, float, float]]):
        arr = np.asarray(q, dtype=float)
        if arr.shape != (4,):
            raise ValueError("Quaternion must be a 4-element array-like")
        self._q = arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Quaternion":
        return cls(arr)

    @classmethod
    def from_euler(cls, pitch: float, roll: float, yaw: float) -> "Quaternion":
        return cls(euler_to_quaternion(roll, pitch, yaw))

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> "Quaternion":
        return cls(rotvec_to_quaternion(rotvec))

    @classmethod
    def from_dcm(cls, dcm: np.ndarray) -> "Quaternion":
        return cls(dir_cos_mat_to_quaternion(dcm))

    def as_ndarray(self) -> np.ndarray:
        return self._q.copy()

    def to_dcm(self) -> np.ndarray:
        return quaternion_to_dir_cos_mat(self._q)

    def to_rotvec(self) -> np.ndarray:
        return quaternion_to_rotvec(self._q)

    def to_euler(self) -> Tuple[float, float, float]:
        return quaternion_to_euler(self._q)

    def conjugate(self) -> "Quaternion":
        return Quaternion(quarternion_conjugate(self._q))

    def normalized(self) -> "Quaternion":
        norm = np.linalg.norm(self._q)
        if norm == 0:
            raise ZeroDivisionError("Cannot normalize a zero-norm quaternion")
        return Quaternion(self._q / norm)

    def __mul__(self, other: Quaternion) -> Quaternion:
        if isinstance(other, Quaternion):
            return Quaternion(quaternion_multiply(self._q, other._q))
        return NotImplemented

    def rotate(self, vector: np.ndarray) -> np.ndarray:
        """用此四元数旋转三维向量。

        :param vector: (3,) ndarray
        :return: 旋转后的向量
        """
        return quaternion_mul_vec(self._q, np.asarray(vector))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return False
        return bool(np.allclose(self._q, other._q))

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array(self._q, dtype=dtype)

    def __repr__(self) -> str:
        q0, q1, q2, q3 = self._q
        return f"Quaternion([{q0:.6g}, {q1:.6g}, {q2:.6g}, {q3:.6g}])"
