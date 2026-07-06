import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from simulation.av_to_imu import av_to_imu
from visualization.plot_utils import plot_trajectory, plot_imu_data, delta_position
from simulation.trajectory_generator import trj_profile
from core.earth_model import EarthModel
from simulation.misalignment_angle import *
from core.basic.angle import *
from core.ins_update import ins_update
from config import wgs84
from core.basic.quaternion import Quaternion
from core.sins_error_transition import build_sins_error_transition
from core.kalman import KalmanFilter
from simulation.imu_adderr import imu_add_errors


def main():
    # trajectory_and_imu_simulation()
    # static_imu_simulation()
    combined_navigation_simulation()


def combined_navigation_simulation():
    """
    组合导航仿真
    """
    # 参数
    n_sub = 2  # 子样数
    imu_dt = 0.1  # imu采样周期
    nav_dt = n_sub * imu_dt  # 导航解算周期
    total_t = 3600.0

    # 初始姿态
    att0 = np.array([0.0, 0.0, 30.0]) * deg_to_rad
    qnb0 = Quaternion.from_euler(*att0)
    # 初始速度和位置
    vn0 = np.zeros(3)
    pos0 = np.array([34.0, 108.0, 100.0]) * \
        np.array([deg_to_rad, deg_to_rad, 1.0])

    # 地球参数
    eth0 = EarthModel(tuple(pos0), vn0)
    # 生成静止 IMU 基准增量
    qb0 = qnb0.conjugate()
    omega_m_base = qb0.rotate(eth0.omega_in_n) * imu_dt
    vm_base = qb0.rotate(-eth0.gravity_n) * imu_dt
    omega_m = np.tile(omega_m_base.reshape(1, 3), (n_sub, 1))
    vm = np.tile(vm_base.reshape(1, 3), (n_sub, 1))

    # 误差参数
    arcmin = min_to_rad
    dph = deg_to_rad / 3600.0
    ug = 1e-6 * wgs84.g0

    phi_err = np.array([0.1, 0.2, 3.0]) * arcmin  # 失准角
    eb = np.array([0.01, 0.015, 0.02]) * dph  # 陀螺零偏
    omega_eb = np.array([0.001, 0.001, 0.001]) * dph  # 角度随机游走
    db = np.array([80.0, 90.0, 100.0]) * ug  # 加速度常值偏值
    wdb = np.array([1.0, 1.0, 1.0]) * ug  # 速度随机游走系数

    # 卡尔曼滤波器初始化
    n_state = 15
    Qk = np.diag(np.hstack([omega_eb, wdb, np.zeros(9)])) ** 2 * nav_dt
    Re = wgs84.a
    rk = np.hstack([[0.1, 0.1, 0.1], np.array([10.0, 10.0]) / Re, [10.0]])
    Rk = np.diag(rk) ** 2
    P0 = np.diag(np.hstack([
        np.array([0.1, 0.1, 10.0]) * deg_to_rad,
        np.ones(3) * 1.0,
        np.array([10.0, 10.0]) / Re,
        10.0,
        np.ones(3) * 0.1 * dph,
        np.ones(3) * 100.0 * ug,
    ])) ** 2
    Hk = np.hstack([np.zeros((6, 3)), np.eye(6), np.zeros((6, 6))])

    kf = KalmanFilter(Qk, Rk, P0, np.zeros((n_state, n_state)), Hk)

    # 准备记录数组
    len_main = int(total_t / imu_dt)
    avp = []
    xkpk = []
    imu_omega_m = []
    imu_vm = []

    # 施加失准角到姿态
    qnb = quarternion_add_misalignment_angle(qnb0, phi_err)
    vn = vn0.copy()
    pos = tuple(pos0)

    t = 0.0
    # 主循环
    for k in range(0, len_main, n_sub):
        t += nav_dt

        # 生成 IMU 子样误差数据
        omega_m_sub, vm_sub = imu_add_errors(
            omega_m, vm, eb, omega_eb, db, wdb, imu_dt)
        imu_omega_m.append(omega_m_sub)
        imu_vm.append(vm_sub)

        # 惯导更新
        qnb, vn, pos = ins_update(qnb, vn, pos, omega_m_sub, vm_sub, imu_dt)

        # 构造误差转移矩阵,状态预测
        eth = EarthModel(pos, vn)
        Cnb = qnb.to_dcm()
        fb = vm_sub.sum(axis=0) / nav_dt
        Ft = build_sins_error_transition(eth, Cnb, fb)
        kf.Phi = np.eye(n_state) + Ft * nav_dt
        kf.update(None, time_meas_both='T')

        # 每秒一次 GPS 量测更新
        if abs(t - round(t)) < nav_dt/2:
            gps = np.hstack([vn0, pos0]) + rk * np.random.randn(6)
            meas = (np.hstack([vn, pos]) - gps).reshape(-1, 1)
            kf.update(meas, time_meas_both='M')
            # 速度反馈
            vn[2] = vn[2] - float(kf.x[5, 0])
            kf.x[5, 0] = 0.0

        dq = qnb * qnb0.conjugate()
        phi_small = -dq.to_rotvec().ravel()
        vn_plot = np.array([vn[1], vn[0], -vn[2]])
        avp.append(
            np.hstack([phi_small, vn_plot, np.array(pos).ravel(), t]))

        # 记录 xkpk: [x(15), diag(P)(15)]
        diagP = np.diag(kf.P)
        xkpk.append(np.hstack([kf.x.ravel(), diagP, t]))

        if (t % 100.0) < nav_dt:
            print(int(t))

    avp = np.asarray(avp)
    xkpk = np.asarray(xkpk)
    imu_omega_m = np.vstack(imu_omega_m)
    imu_vm = np.vstack(imu_vm)

    # 绘图（保存在当前工作目录）
    fig1 = plot_trajectory(avp[:, -1], avp[:, 0:3], avp[:, 3:6], avp[:, 6:9])
    fig1.savefig("combined_trajectory.png", dpi=120, bbox_inches="tight")
    fig2 = plot_imu_data(np.linspace(
        0, total_t, imu_omega_m.shape[0] + 1), imu_omega_m, imu_vm, imu_dt)
    fig2.savefig("combined_imu.png", dpi=120, bbox_inches="tight")

    try:
        from visualization.plot_utils import plot_compare_avp_xkpk

        fig3 = plot_compare_avp_xkpk(avp, xkpk)
        if fig3 is not None:
            fig3.savefig("combined_compare.png", dpi=120, bbox_inches="tight")
    except Exception:
        pass

    print("Combined navigation simulation finished and figures saved.")


def trajectory_and_imu_simulation():
    # Simulation parameters
    dt = 0.01  # Sample interval (seconds)

    # Initial conditions
    att0 = np.array([0.0, 0.0, 90.0]) * deg_to_rad  # pitch, roll, yaw (rad)
    vn0 = np.array([0.0, 0.0, 0.0])  # north, east, down velocity (m/s)
    pos0 = np.array([34.0, 108.0, 100.0]) * \
        np.array([deg_to_rad, deg_to_rad, 1.0])  # lat, lon (rad), h (m)

    # Maneuver sequence: [wx, wy, wz (deg/s), a_forward (m/s²), duration (s)]
    maneuvers_deg = np.array([
        [0.0,   0.0,   0.0,   0.0,  10.0],   # stationary
        [0.0,   0.0,   0.0,   1.0,  10.0],   # accelerate
        [0.0,   0.0,   0.0,   0.0,  10.0],   # constant velocity
        [5.0,   0.0,   0.0,   0.0,   4.0],   # pitch up
        [0.0,   0.0,   0.0,   0.0,  10.0],   # constant velocity
        [-5.0,  0.0,   0.0,   0.0,   4.0],   # pitch down
        [0.0,   0.0,   0.0,   0.0,  10.0],   # constant velocity
        [0.0,  10.0,   0.0,   0.0,   1.0],   # roll
        [0.0,   0.0,   9.0,   0.0,  10.0],   # turn
        [0.0, -10.0,   0.0,   0.0,   1.0],   # roll back
        [0.0,   0.0,   0.0,   0.0,  10.0],   # constant velocity
        [0.0,   0.0,   0.0,  -1.0,  10.0],   # decelerate
        [0.0,   0.0,   0.0,   0.0,  10.0],   # stationary
    ])

    # Convert angular rates from deg/s to rad/s
    maneuvers = maneuvers_deg.copy()
    maneuvers[:, 0:3] = maneuvers_deg[:, 0:3] * deg_to_rad

    print(f"Simulating trajectory with dt={dt}s...")

    # Generate trajectory
    att, vn, pos = trj_profile(att0, vn0, pos0, maneuvers, dt)
    print(f"  Generated {len(att)} attitude samples")

    # Generate IMU data
    wm, vm = av_to_imu(att, vn, pos, dt)
    print(f"  Generated {len(wm)} IMU samples")

    # Time vector
    time = np.arange(len(att)) * dt

    # Plot trajectory
    print("Plotting trajectory...")
    fig1 = plot_trajectory(time, att, vn, pos, deg_to_rad)
    fig1.savefig("trajectory.png", dpi=100, bbox_inches="tight")
    print("  -> saved trajectory.png")

    # Plot IMU data
    print("Plotting IMU data...")
    fig2 = plot_imu_data(time, wm, vm, dt, deg_to_rad)
    fig2.savefig("imu_data.png", dpi=100, bbox_inches="tight")
    print("  -> saved imu_data.png")

    print("\nSimulation complete!")
    print(
        f"Final position: lat={pos[-1, 0]*rad_to_deg:.4f}°, lon={pos[-1, 1]*rad_to_deg:.4f}°, h={pos[-1, 2]:.2f}m")

    # Show plots (optional; comment out to disable interactive display)
    plt.show()


def static_imu_simulation():
    dt = 0.1  # 采样间隔
    n_sub = 2  # 子样数
    total_t = 3600

    att0 = np.array([0.0, 0.0, 30.0]) * deg_to_rad
    vel0 = np.zeros(3)
    pos0 = np.array([34.0, 108.0, 100.0]) * \
        np.array([deg_to_rad, deg_to_rad, 1.0])
    pos0 = tuple(pos0)
    quat_n_b = Quaternion.from_euler(*att0)

    # 生成静止的imu数据
    quat_b_n = quat_n_b.conjugate()
    earth = EarthModel(pos0, vel0)
    delta_theta = quat_b_n.rotate(earth.omega_in_n) * dt
    delta_v = quat_b_n.rotate(-earth.gravity_n) * dt
    delta_theta = np.tile(delta_theta, (n_sub, 1))
    delta_v = np.tile(delta_v, (n_sub, 1))

    # 加入失准角
    misalignment = np.array([0.1, 0.2, 3.0]) * min_to_rad
    quat_n_b_misaligned = quarternion_add_misalignment_angle(
        quat_n_b, misalignment)

    # 计算惯导更新
    total_samples = int(total_t / dt)
    total_steps = total_samples // n_sub
    results = []
    t = 0.0
    quat, vel, pos = quat_n_b_misaligned, vel0, pos0

    start_time = time.time()
    report_every = max(1, total_steps // 10)

    for step in range(1, total_steps + 1):
        t += dt * n_sub
        quat, vel, pos = ins_update(quat, vel, pos, delta_theta, delta_v, dt)
        vel[2] = 0.0
        results.append([t, *quat.to_euler(), *vel, *pos])

        # 进度打印
        if step % report_every == 0 or step == total_steps:
            elapsed = time.time() - start_time
            percent = (step / total_steps *
                       100.0) if total_steps > 0 else 100.0
            eta = elapsed * (total_steps / step -
                             1) if step < total_steps else 0.0
            print(
                f"Progress: {percent:.1f}% ({step}/{total_steps}) elapsed {elapsed:.1f}s ETA {eta:.1f}s")

    df = pd.DataFrame(
        results, columns=["time", "pitch", "roll", "yaw",
                          "vn", "ve", "vd", "lat", "lon", "h"]
    )
    df[['pitch', 'roll', 'yaw']] *= rad_to_deg

    # 画图
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].plot(df.time, df[['pitch', 'roll']])
    ax[0, 0].set_ylabel("θ, γ/°")
    ax[0, 0].legend(["θ", "γ"])

    ax[0, 1].plot(df.time, df.yaw)
    ax[0, 1].set_ylabel("ψ/°")
    ax[0, 1].legend(["ψ"])
    ax[0, 1].ticklabel_format(useOffset=False, style='plain')

    ax[1, 0].plot(df.time, df[['vn', 've', 'vd']])
    ax[1, 0].set_ylabel("V (m/s)")
    ax[1, 0].legend(["Vn", "Ve", "Vd"])

    del_pos = delta_position(df[['lat', 'lon', 'h']].values)
    ax[1, 1].plot(df.time, del_pos)
    ax[1, 1].set_ylabel("Δp/m")
    ax[1, 1].legend(["ΔN", "ΔE", "Δh"])

    fig.suptitle("Static IMU Simulation")
    fig.savefig("static_imu_simulation.png", dpi=100, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
