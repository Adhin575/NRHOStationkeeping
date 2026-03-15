# ==========================================
# VISUALIZATION CODE
# Animated True vs Estimated (Fusion)
# ==========================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mu = 0.0121505856

def cr3bp(t, state):
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
    return [vx, vy, ax, ay]

def gnss_measurement(x):
    sat_pos = np.array([0.0, 0.0])
    return np.array([np.linalg.norm(x[:2] - sat_pos)])

def optical_measurement(x):
    moon_pos = np.array([1 - mu, 0])
    los = moon_pos - x[:2]
    return los / np.linalg.norm(los)

def numerical_F(x, dt):
    eps = 1e-6
    F = np.zeros((4,4))
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        f1 = np.array(cr3bp(0, x + dx))
        f2 = np.array(cr3bp(0, x - dx))
        F[:,i] = (f1 - f2) / (2*eps)
    return np.eye(4) + F*dt

dt = 0.01
t_final = 5
time = np.arange(0, t_final, dt)

x_true = np.array([1.05, 0.0, 0.0, 0.15])
x_est  = x_true + np.array([0.01, -0.01, 0.01, -0.01])

P = np.eye(4) * 1e-3
Q = np.eye(4) * 1e-6
R_g = np.array([[1e-4]])
R_o = np.eye(2) * 1e-6

true_states = []
est_states = []

for t in time:

    sol = solve_ivp(cr3bp, [0, dt], x_true)
    x_true = sol.y[:, -1]

    F = numerical_F(x_est, dt)
    sol_est = solve_ivp(cr3bp, [0, dt], x_est)
    x_pred = sol_est.y[:, -1]
    P = F @ P @ F.T + Q

    # GNSS
    z = gnss_measurement(x_true) + np.random.normal(0, 1e-2, 1)
    h = gnss_measurement(x_pred)
    H = np.zeros((1,4))
    eps = 1e-6
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        H[0,i] = ((gnss_measurement(x_pred + dx) - h) / eps).item()
    y = z - h
    S = H @ P @ H.T + R_g
    K = P @ H.T @ np.linalg.inv(S)
    x_pred = x_pred + (K @ y).flatten()
    P = (np.eye(4) - K @ H) @ P

    # Optical
    z = optical_measurement(x_true) + np.random.multivariate_normal([0,0], R_o)
    h = optical_measurement(x_pred)
    H = np.zeros((2,4))
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        H[:,i] = (optical_measurement(x_pred + dx) - h) / eps
    y = z - h
    S = H @ P @ H.T + R_o
    K = P @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P = (np.eye(4) - K @ H) @ P

    true_states.append(x_true.copy())
    est_states.append(x_est.copy())

true_states = np.array(true_states)
est_states = np.array(est_states)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0.93, 1.07)
ax.set_ylim(-0.06, 0.06)
ax.set_title("True vs Estimated Trajectory (Fusion)")
ax.set_xlabel("x")
ax.set_ylabel("y")

line_true, = ax.plot([], [])
line_est,  = ax.plot([], [])

def update(frame):
    line_true.set_data(true_states[:frame,0], true_states[:frame,1])
    line_est.set_data(est_states[:frame,0], est_states[:frame,1])
    return line_true, line_est

ani = animation.FuncAnimation(fig, update, frames=len(true_states), interval=20)
plt.show()
