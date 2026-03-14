"""
Stable / Unstable Eigen-Structure Geometry Diagram
NRHO orbit with Floquet eigenvectors and manifold-aligned control projection.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# ── CR3BP ─────────────────────────────────────────────────────────────────────
MU = 0.0121505856

def cr3bp_eom(t, s):
    x,y,z,vx,vy,vz = s
    r1 = np.sqrt((x+MU)**2      + y**2 + z**2)
    r2 = np.sqrt((x-1+MU)**2   + y**2 + z**2)
    ax = 2*vy + x  - (1-MU)*(x+MU)/r1**3 - MU*(x-1+MU)/r2**3
    ay =-2*vx + y  - (1-MU)*y/r1**3      - MU*y/r2**3
    az =            -(1-MU)*z/r1**3      - MU*z/r2**3
    return [vx,vy,vz,ax,ay,az]

def cr3bp_jac(s):
    x,y,z = s[:3]
    r1 = np.sqrt((x+MU)**2    +y**2+z**2)
    r2 = np.sqrt((x-1+MU)**2  +y**2+z**2)
    m1,m2 = 1-MU, MU
    Uxx = 1 - m1/r1**3 + 3*m1*(x+MU)**2/r1**5    - m2/r2**3 + 3*m2*(x-1+MU)**2/r2**5
    Uyy = 1 - m1/r1**3 + 3*m1*y**2/r1**5          - m2/r2**3 + 3*m2*y**2/r2**5
    Uzz =   - m1/r1**3 + 3*m1*z**2/r1**5          - m2/r2**3 + 3*m2*z**2/r2**5
    Uxy = 3*m1*(x+MU)*y/r1**5  + 3*m2*(x-1+MU)*y/r2**5
    Uxz = 3*m1*(x+MU)*z/r1**5  + 3*m2*(x-1+MU)*z/r2**5
    Uyz = 3*m1*y*z/r1**5       + 3*m2*y*z/r2**5
    return np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],
                     [Uxx,Uxy,Uxz,0,2,0],[Uxy,Uyy,Uyz,-2,0,0],[Uxz,Uyz,Uzz,0,0,0]])

# ── Propagate NRHO (L2 southern) ──────────────────────────────────────────────
s0 = np.array([1.0213448959, 0.0, -0.1818450156, 0.0, -0.1036971744, 0.0])
T  = 1.5111
t_eval = np.linspace(0, T, 600)
sol = solve_ivp(cr3bp_eom, (0,T), s0, method='DOP853',
                t_eval=t_eval, rtol=1e-12, atol=1e-12)
orbit = sol.y.T   # (600,6)

# ── STM → monodromy → eigenvectors ───────────────────────────────────────────
def stm_eom(t, phi_flat):
    s   = phi_flat[:6]
    phi = phi_flat[6:].reshape(6,6)
    return np.concatenate([cr3bp_eom(t,s), (cr3bp_jac(s)@phi).flatten()])

stm_sol   = solve_ivp(stm_eom,(0,T), np.concatenate([s0,np.eye(6).flatten()]),
                      method='DOP853', rtol=1e-12, atol=1e-12)
M         = stm_sol.y[6:,-1].reshape(6,6)
evals, evecs = np.linalg.eig(M)
mags      = np.abs(evals)

i_u = np.argmax(mags)
i_s = np.argmin(mags)
v_u = evecs[:,i_u].real;  v_u /= np.linalg.norm(v_u)
v_s = evecs[:,i_s].real;  v_s /= np.linalg.norm(v_s)

lam_u = mags[i_u]
lam_s = mags[i_s]

print(f"|λ_u| = {lam_u:.4f},  |λ_s| = {lam_s:.4f}")

# ── Choose a base point on the orbit (periapsis = min |r_moon|) ───────────────
r_moon = np.sqrt((orbit[:,0]-(1-MU))**2 + orbit[:,1]**2 + orbit[:,2]**2)
k      = np.argmin(r_moon)          # periapsis (near Moon)
p      = orbit[k,:3]                # position only

# ── Arrow scale relative to orbit extent ─────────────────────────────────────
orb_scale = np.ptp(orbit[:,2])      # z-range of orbit
L  = orb_scale * 0.22               # arrow length

# Perturb vector: slightly off-axis in 3D
delta_hat = np.array([0.3, 0.5, 0.8]); delta_hat /= np.linalg.norm(delta_hat)
dv_u      = np.dot(delta_hat, v_u[:3]) * v_u[:3]
dv_s      = np.dot(delta_hat, v_s[:3]) * v_s[:3]
dv_u     /= (np.linalg.norm(dv_u)+1e-12)
dv_s     /= (np.linalg.norm(dv_s)+1e-12)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 8.5), facecolor="#080c18")
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor("#080c18")
fig.patch.set_facecolor("#080c18")

# ── NRHO orbit curve with depth-faded color ───────────────────────────────────
z_norm = (orbit[:,2] - orbit[:,2].min()) / np.ptp(orbit[:,2])
for i in range(len(orbit)-1):
    alpha = 0.35 + 0.55*z_norm[i]
    ax.plot(orbit[i:i+2,0], orbit[i:i+2,1], orbit[i:i+2,2],
            color="#38bdf8", lw=1.4, alpha=alpha)

# ── Base point ────────────────────────────────────────────────────────────────
ax.scatter(*p, color="white", s=38, zorder=10)

# helper: 3D quiver with clean style
def arrow3d(ax, origin, direction, length, color, lw=2.2, alpha=1.0,
            mutation_scale=14):
    o = np.array(origin)
    d = np.array(direction)
    ax.quiver(*o, *(d*length), color=color, linewidth=lw, alpha=alpha,
              arrow_length_ratio=0.22, normalize=False)

# ── Unstable eigenvector (red) ────────────────────────────────────────────────
arrow3d(ax, p,  v_u[:3], L,     "#ef4444", lw=2.5)
arrow3d(ax, p, -v_u[:3], L*0.5, "#ef4444", lw=1.3, alpha=0.35)  # ghost tail
ax.text(*(p + v_u[:3]*L*1.18), r"$\mathbf{v}_u$",
        color="#ef4444", fontsize=14, fontweight="bold",
        ha="center", va="center")

# ── Stable eigenvector (blue) ─────────────────────────────────────────────────
arrow3d(ax, p,  v_s[:3], L,     "#3b82f6", lw=2.5)
arrow3d(ax, p, -v_s[:3], L*0.5, "#3b82f6", lw=1.3, alpha=0.35)
ax.text(*(p + v_s[:3]*L*1.18), r"$\mathbf{v}_s$",
        color="#3b82f6", fontsize=14, fontweight="bold",
        ha="center", va="center")

# ── Perturbation arrow (white, dashed-style via short segments) ───────────────
pert_dir = delta_hat * 0.6 + 0.4*v_u[:3]
pert_dir /= np.linalg.norm(pert_dir)
pert_len  = L * 0.80
pert_end  = p + pert_dir * pert_len
# Dashed: alternating segments
n_dash = 8
for i in range(n_dash):
    if i % 2 == 0:
        t0,t1 = i/n_dash, (i+0.45)/n_dash
        seg0  = p + pert_dir * pert_len * t0
        seg1  = p + pert_dir * pert_len * t1
        ax.plot(*zip(seg0,seg1), color="#e2e8f0", lw=1.6, alpha=0.85)
ax.quiver(*pert_end, *(pert_dir*0.001),   # tiny quiver just for arrowhead
          color="#e2e8f0", lw=1.6, arrow_length_ratio=80, normalize=False)
ax.text(*(pert_end + pert_dir*L*0.18), r"$\delta\mathbf{x}$",
        color="#e2e8f0", fontsize=11, ha="center")

# ── Decomposition: project δx onto v_u and v_s ───────────────────────────────
# Show the two components as thin colored dashed lines from base point
comp_u = np.dot(pert_dir, v_u[:3]) * v_u[:3] * pert_len
comp_s = np.dot(pert_dir, v_s[:3]) * v_s[:3] * pert_len

for comp, col in [(comp_u,"#ef4444"),(comp_s,"#3b82f6")]:
    end = p + comp
    ax.plot(*zip(p,end), color=col, lw=1.1, linestyle=":", alpha=0.55)

# ── Control thrust arrow (green, aligned with -v_u to cancel unstable mode) ──
ctrl_dir  = -v_u[:3].copy()
ctrl_len  = L * 0.72
ctrl_start = p + pert_dir * pert_len * 0.5   # starts from midpoint of perturbation
arrow3d(ax, ctrl_start, ctrl_dir, ctrl_len, "#22c55e", lw=2.3, alpha=0.95)
ax.text(*(ctrl_start + ctrl_dir*ctrl_len*1.25),
        "Control\nprojection\nonto $v_s$ subspace",
        color="#22c55e", fontsize=8.5, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a1a0f",
                  edgecolor="#16a34a", lw=0.9, alpha=0.88))

# ── Invariant subspace planes (translucent) ───────────────────────────────────
def draw_subspace_plane(ax, origin, n1, n2, size, color, alpha=0.07):
    """Draw a parallelogram spanned by n1, n2 centered at origin."""
    corners = np.array([
        origin - n1*size - n2*size,
        origin + n1*size - n2*size,
        origin + n1*size + n2*size,
        origin - n1*size + n2*size,
    ])
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection([corners], alpha=alpha, facecolor=color,
                             edgecolor=color, linewidth=0.5)
    ax.add_collection3d(poly)

# Unstable subspace plane (spanned by v_u and a tangent)
tangent = orbit[k+5,:3] - orbit[k,:3]; tangent /= np.linalg.norm(tangent)
draw_subspace_plane(ax, p, v_u[:3]*0.8, tangent*0.5, L*1.2, "#ef4444", alpha=0.08)

# Stable subspace plane
draw_subspace_plane(ax, p, v_s[:3]*0.8, tangent*0.5, L*1.2, "#3b82f6", alpha=0.08)

# ── Floquet multiplier annotation ─────────────────────────────────────────────
ann_pos = p + np.array([0, 0, L*1.6])
ax.text(*ann_pos,
        f"|λ_u| = {lam_u:.2f}  (diverges)\n|λ_s| = {lam_s:.4f}  (converges)",
        color="#94a3b8", fontsize=8.5, ha="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a",
                  edgecolor="#334155", lw=0.9, alpha=0.90))

# ── Moon (small sphere at correct position) ───────────────────────────────────
u_s = np.linspace(0,2*np.pi,20); v_s2= np.linspace(0,np.pi,15)
moon_r = 0.012
mx = (1-MU) + moon_r*np.outer(np.cos(u_s),np.sin(v_s2))
my =           moon_r*np.outer(np.sin(u_s),np.sin(v_s2))
mz =           moon_r*np.outer(np.ones_like(u_s),np.cos(v_s2))
ax.plot_surface(mx,my,mz, color="#9e9e9e", alpha=0.55, zorder=1, linewidth=0)
ax.text(1-MU, 0, -0.025, "Moon", color="#bdbdbd", fontsize=7.5, ha="center")

# ── Axes style ────────────────────────────────────────────────────────────────
ax.set_xlabel("x  (EM units)", color="#475569", fontsize=8, labelpad=6)
ax.set_ylabel("y  (EM units)", color="#475569", fontsize=8, labelpad=6)
ax.set_zlabel("z  (EM units)", color="#475569", fontsize=8, labelpad=6)
ax.tick_params(colors="#334155", labelsize=7)
ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("#111827")
ax.yaxis.pane.set_edgecolor("#111827")
ax.zaxis.pane.set_edgecolor("#111827")
ax.grid(True, color="#111827", lw=0.5, alpha=0.5)

# View angle: show orbit shape + eigenvectors clearly
ax.view_init(elev=22, azim=-55)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color="#38bdf8", label="NRHO orbit"),
    mpatches.Patch(color="#ef4444", label=r"$\mathbf{v}_u$ — unstable eigenvector  (|λ|>1)"),
    mpatches.Patch(color="#3b82f6", label=r"$\mathbf{v}_s$ — stable eigenvector  (|λ|<1)"),
    mpatches.Patch(color="#e2e8f0", label=r"$\delta\mathbf{x}$ — perturbation"),
    mpatches.Patch(color="#22c55e", label="Control thrust (manifold-aligned)"),
]
ax.legend(handles=legend_handles,
          loc="upper left", fontsize=8.5,
          framealpha=0.88, facecolor="#0f172a",
          edgecolor="#1e293b", labelcolor="white",
          borderpad=0.8, handlelength=1.5)

ax.set_title("NRHO Floquet Eigen-Structure  ·  Stable / Unstable Invariant Subspaces",
             color="white", fontsize=13, fontweight="bold", pad=14)

plt.tight_layout()
plt.show()
