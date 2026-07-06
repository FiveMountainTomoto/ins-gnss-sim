import numpy as np

from core.basic import linalg
from core.basic.quaternion import Quaternion


def test_skew_symmetric():
    v = np.array([1.0, 2.0, 3.0])
    S = linalg.skew_symmetric(v)
    assert S.shape == (3, 3)
    # S * v should be zero vector
    res = S @ v
    assert np.allclose(res, np.zeros(3))


def test_euler_quaternion_roundtrip():
    roll, pitch, yaw = 0.12, -0.34, 1.0
    q = linalg.euler_to_quaternion(roll, pitch, yaw)
    rpy = linalg.quaternion_to_euler(q)
    assert np.allclose([roll, pitch, yaw], rpy, atol=1e-6)


def test_rotvec_quaternion_roundtrip():
    rotvec = np.array([0.2, -0.1, 0.05])
    q = linalg.rotvec_to_quaternion(rotvec)
    rv = linalg.quaternion_to_rotvec(q)
    assert np.allclose(rotvec, rv, atol=1e-6)


def test_quaternion_class_rotate_and_tuple():
    q = Quaternion.from_euler(0.1, 0.2, 0.3)
    t = q.as_ndarray()
    assert isinstance(t, np.ndarray) and len(t) == 4
    v = np.array([1.0, 0.0, 0.0])
    vr = q.rotate(v)
    # rotated vector should have same norm
    assert np.isclose(np.linalg.norm(v), np.linalg.norm(vr))


def test_quaternion_mul_and_inverse():
    q1 = Quaternion.from_euler(0.1, 0.2, 0.3)
    q2 = Quaternion.from_euler(-0.2, 0.05, 0.4)
    q12 = q1 * q2
    # for unit quaternions inverse == conjugate, use conjugate here
    conj = q2.conjugate()
    res = q12 * conj
    # res should equal q1 (within tolerance)
    assert np.allclose(res.as_ndarray(), q1.as_ndarray(),  # type: ignore
                       atol=1e-6)


def test_quaternion_normalize():
    # 构造一个非单位四元数
    q = Quaternion(np.array([2.0, 0.0, 0.0, 0.0]))

    # normalized() 返回新对象，且范数为1，原对象不变
    qn = q.normalized()
    assert np.isclose(np.linalg.norm(qn.as_ndarray()), 1.0)
    assert np.isclose(np.linalg.norm(q.as_ndarray()), 2.0)

    # 归一化结果应为 [1,0,0,0]
    assert np.allclose(qn.as_ndarray(), np.array([1.0, 0.0, 0.0, 0.0]))
