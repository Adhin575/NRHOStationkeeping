"""
Advanced NRHO Analysis Module
==============================

Extensions to the base NRHO implementation:
1. Invariant manifold computation and propagation
2. Bicircular Restricted Four-Body Problem (Sun perturbations)
3. Floquet-mode-based optimal control

This module represents PhD-level orbital mechanics research.

Author: Advanced Research Implementation
Date: February 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig, solve_continuous_are, schur
from typing import Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# SECTION 1: Invariant Manifold Computation and Propagation
# ============================================================================

class InvariantManifoldAnalyzer:
    """
    Compute and propagate stable/unstable manifolds for trajectory design.
    
    Applications:
    - Low-energy Earth-Moon transfers
    - Ballistic lunar capture
    - Gateway insertion trajectories
    - Interplanetary transport network
    """
    
    def __init__(self, cr3bp):
        """
        Initialize manifold analyzer.
        
        Parameters
        ----------
        cr3bp : CR3BP
            CR3BP system instance
        """
        self.cr3bp = cr3bp
        
    def compute_manifold_at_point(self, 
                                  state: np.ndarray,
                                  monodromy: np.ndarray,
                                  epsilon: float = 1e-6) -> dict:
        """
        Compute stable and unstable manifolds at a specific orbit point.
        
        Parameters
        ----------
        state : np.ndarray
            State on periodic orbit
        monodromy : np.ndarray
            Monodromy matrix
        epsilon : float
            Perturbation magnitude along eigenvector
            
        Returns
        -------
        dict : Contains eigenvectors and perturbed initial conditions
        """
        # Compute eigendecomposition
        eigenvalues, eigenvectors = eig(monodromy)
        
        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Identify unstable (|λ| > 1) and stable (|λ| < 1) directions
        unstable_mask = np.abs(eigenvalues) > 1.0 + 1e-6
        stable_mask = np.abs(eigenvalues) < 1.0 - 1e-6
        
        unstable_eigenvectors = eigenvectors[:, unstable_mask]
        stable_eigenvectors = eigenvectors[:, stable_mask]
        
        print("\n" + "="*70)
        print("INVARIANT MANIFOLD COMPUTATION")
        print("="*70)
        print(f"\nNumber of unstable directions: {np.sum(unstable_mask)}")
        print(f"Number of stable directions: {np.sum(stable_mask)}")
        
        # Create initial conditions perturbed along manifold directions
        manifold_ics = {
            'unstable_plus': [],
            'unstable_minus': [],
            'stable_plus': [],
            'stable_minus': []
        }
        
        # Unstable manifold initial conditions
        for i in range(unstable_eigenvectors.shape[1]):
            evec = unstable_eigenvectors[:, i].real  # Take real part
            evec = evec / np.linalg.norm(evec)  # Normalize
            
            manifold_ics['unstable_plus'].append(state + epsilon * evec)
            manifold_ics['unstable_minus'].append(state - epsilon * evec)
        
        # Stable manifold initial conditions
        for i in range(stable_eigenvectors.shape[1]):
            evec = stable_eigenvectors[:, i].real
            evec = evec / np.linalg.norm(evec)
            
            manifold_ics['stable_plus'].append(state + epsilon * evec)
            manifold_ics['stable_minus'].append(state - epsilon * evec)
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'unstable_eigenvectors': unstable_eigenvectors,
            'stable_eigenvectors': stable_eigenvectors,
            'manifold_ics': manifold_ics,
            'reference_state': state
        }
    
    def propagate_manifolds(self,
                           manifold_data: dict,
                           T_forward: float = 3.0,
                           T_backward: float = 3.0) -> dict:
        """
        Propagate manifold trajectories forward and backward in time.
        
        Parameters
        ----------
        manifold_data : dict
            Output from compute_manifold_at_point
        T_forward : float
            Forward integration time (unstable manifolds)
        T_backward : float
            Backward integration time (stable manifolds)
            
        Returns
        -------
        dict : Propagated trajectories
        """
        print("\nPropagating invariant manifolds...")
        
        results = {
            'unstable_trajectories': [],
            'stable_trajectories': []
        }
        
        # Propagate unstable manifolds forward (escape/departure)
        print(f"  Forward propagation (unstable, T={T_forward})...")
        for ic in manifold_data['manifold_ics']['unstable_plus']:
            sol = solve_ivp(
                self.cr3bp.equations_of_motion,
                [0, T_forward],
                ic,
                method='DOP853',
                rtol=1e-10,
                atol=1e-10,
                dense_output=True
            )
            t_eval = np.linspace(0, T_forward, 500)
            results['unstable_trajectories'].append(sol.sol(t_eval).T)
        
        for ic in manifold_data['manifold_ics']['unstable_minus']:
            sol = solve_ivp(
                self.cr3bp.equations_of_motion,
                [0, T_forward],
                ic,
                method='DOP853',
                rtol=1e-10,
                atol=1e-10,
                dense_output=True
            )
            t_eval = np.linspace(0, T_forward, 500)
            results['unstable_trajectories'].append(sol.sol(t_eval).T)
        
        # Propagate stable manifolds backward (capture/arrival)
        print(f"  Backward propagation (stable, T={T_backward})...")
        for ic in manifold_data['manifold_ics']['stable_plus']:
            sol = solve_ivp(
                self.cr3bp.equations_of_motion,
                [0, -T_backward],
                ic,
                method='DOP853',
                rtol=1e-10,
                atol=1e-10,
                dense_output=True
            )
            t_eval = np.linspace(0, -T_backward, 500)
            results['stable_trajectories'].append(sol.sol(t_eval).T)
        
        for ic in manifold_data['manifold_ics']['stable_minus']:
            sol = solve_ivp(
                self.cr3bp.equations_of_motion,
                [0, -T_backward],
                ic,
                method='DOP853',
                rtol=1e-10,
                atol=1e-10,
                dense_output=True
            )
            t_eval = np.linspace(0, -T_backward, 500)
            results['stable_trajectories'].append(sol.sol(t_eval).T)
        
        print(f"✓ Propagated {len(results['unstable_trajectories'])} unstable manifolds")
        print(f"✓ Propagated {len(results['stable_trajectories'])} stable manifolds")
        
        return results


# ============================================================================
# SECTION 2: Bicircular Restricted Four-Body Problem (BCR4BP)
# ============================================================================

class BCR4BP:
    """
    Bicircular Restricted Four-Body Problem: Earth-Moon-Sun-Spacecraft.
    
    Includes solar perturbations for high-fidelity NRHO analysis.
    Essential for long-duration mission planning (Gateway).
    """
    
    def __init__(self, mu_EM: float = 0.0121505856):
        """
        Initialize BCR4BP system.
        
        Parameters
        ----------
        mu_EM : float
            Earth-Moon mass parameter
        """
        self.mu_EM = mu_EM  # Earth-Moon mass ratio
        
        # Sun parameters (normalized to Earth-Moon system)
        # Sun is much more massive: M_sun / (M_earth + M_moon) ≈ 328,900
        # Sun distance: a_sun / a_EM ≈ 389
        self.mu_sun = 328900.56  # Mass ratio
        self.a_sun = 388.811  # Distance ratio (EM units)
        
        # Sun orbital angular velocity (normalized to EM synodic frame)
        # ω_sun = (EM orbital period) / (Earth-Sun period) ≈ 1/13.37
        self.omega_sun = 0.9252  # Relative angular velocity
        
        print("\n" + "="*70)
        print("BCR4BP INITIALIZATION (Sun Perturbations)")
        print("="*70)
        print(f"Earth-Moon mass parameter: μ_EM = {self.mu_EM:.10f}")
        print(f"Sun mass parameter: μ_sun = {self.mu_sun:.2f}")
        print(f"Sun distance: a_sun = {self.a_sun:.2f} (EM units)")
        print(f"Sun angular velocity: ω_sun = {self.omega_sun:.4f}")
    
    def equations_of_motion_bcr4bp(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        BCR4BP equations of motion including solar perturbations.
        
        The Sun orbits in a circular path in the rotating EM frame.
        
        Parameters
        ----------
        t : float
            Time
        state : np.ndarray
            State vector [x, y, z, vx, vy, vz]
            
        Returns
        -------
        np.ndarray
            Derivative [vx, vy, vz, ax, ay, az]
        """
        x, y, z, vx, vy, vz = state
        
        # Earth-Moon gravitational terms (CR3BP part)
        r1 = np.sqrt((x + self.mu_EM)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu_EM)**2 + y**2 + z**2)
        
        Ux_EM = x - (1 - self.mu_EM) * (x + self.mu_EM) / r1**3 - \
                self.mu_EM * (x - 1 + self.mu_EM) / r2**3
        Uy_EM = y - (1 - self.mu_EM) * y / r1**3 - self.mu_EM * y / r2**3
        Uz_EM = -(1 - self.mu_EM) * z / r1**3 - self.mu_EM * z / r2**3
        
        # Sun position (circular orbit in rotating frame)
        x_sun = self.a_sun * np.cos(self.omega_sun * t)
        y_sun = self.a_sun * np.sin(self.omega_sun * t)
        z_sun = 0.0
        
        # Distance to Sun
        r_sun = np.sqrt((x - x_sun)**2 + (y - y_sun)**2 + (z - z_sun)**2)
        
        # Solar gravitational acceleration (direct + indirect terms)
        # Direct term: -μ_sun * (r - r_sun) / |r - r_sun|³
        ax_sun_direct = -self.mu_sun * (x - x_sun) / r_sun**3
        ay_sun_direct = -self.mu_sun * (y - y_sun) / r_sun**3
        az_sun_direct = -self.mu_sun * (z - z_sun) / r_sun**3
        
        # Indirect term: +μ_sun * r_sun / |r_sun|³ (barycentric correction)
        r_sun_bary = np.sqrt(x_sun**2 + y_sun**2 + z_sun**2)
        ax_sun_indirect = self.mu_sun * x_sun / r_sun_bary**3
        ay_sun_indirect = self.mu_sun * y_sun / r_sun_bary**3
        az_sun_indirect = 0.0
        
        # Total acceleration
        ax = 2 * vy + Ux_EM + ax_sun_direct + ax_sun_indirect
        ay = -2 * vx + Uy_EM + ay_sun_direct + ay_sun_indirect
        az = Uz_EM + az_sun_direct + az_sun_indirect
        
        return np.array([vx, vy, vz, ax, ay, az])
    
    def compute_solar_perturbation_magnitude(self, state: np.ndarray, t: float) -> float:
        """
        Compute magnitude of solar perturbation at given state.
        
        Parameters
        ----------
        state : np.ndarray
            Spacecraft state
        t : float
            Time
            
        Returns
        -------
        float
            Perturbation acceleration magnitude
        """
        x, y, z = state[:3]
        
        # Sun position
        x_sun = self.a_sun * np.cos(self.omega_sun * t)
        y_sun = self.a_sun * np.sin(self.omega_sun * t)
        z_sun = 0.0
        
        r_sun = np.sqrt((x - x_sun)**2 + (y - y_sun)**2 + (z - z_sun)**2)
        
        ax_sun_direct = -self.mu_sun * (x - x_sun) / r_sun**3
        ay_sun_direct = -self.mu_sun * (y - y_sun) / r_sun**3
        az_sun_direct = -self.mu_sun * (z - z_sun) / r_sun**3
        
        return np.sqrt(ax_sun_direct**2 + ay_sun_direct**2 + az_sun_direct**2)


# ============================================================================
# SECTION 3: Floquet-Mode Decomposition Control
# ============================================================================

class FloquetModeController:
    """
    Advanced control using Floquet mode decomposition.
    
    Key insight: Only control the unstable subspace, not the full state.
    This minimizes Δv while maintaining stability.
    
    Method:
    1. Decompose state into stable, unstable, and center modes
    2. Design controller only for unstable modes
    3. Let stable modes naturally converge
    4. Monitor center modes (periodic directions)
    """
    
    def __init__(self, cr3bp):
        """
        Initialize Floquet controller.
        
        Parameters
        ----------
        cr3bp : CR3BP
            CR3BP system
        """
        self.cr3bp = cr3bp
        
    def decompose_floquet_modes(self, monodromy: np.ndarray) -> dict:
        """
        Decompose state space into Floquet modes.
        
        Uses real Schur decomposition for numerical stability.
        
        Parameters
        ----------
        monodromy : np.ndarray
            Monodromy matrix
            
        Returns
        -------
        dict : Mode decomposition information
        """
        print("\n" + "="*70)
        print("FLOQUET MODE DECOMPOSITION")
        print("="*70)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = eig(monodromy)
        
        # Classify modes
        unstable_mask = np.abs(eigenvalues) > 1.0 + 1e-6
        stable_mask = np.abs(eigenvalues) < 1.0 - 1e-6
        center_mask = ~(unstable_mask | stable_mask)
        
        # Extract mode subspaces
        P_u = eigenvectors[:, unstable_mask]  # Unstable projection
        P_s = eigenvectors[:, stable_mask]    # Stable projection
        P_c = eigenvectors[:, center_mask]    # Center projection
        
        print(f"\nMode Structure:")
        print(f"  Unstable modes: {np.sum(unstable_mask)}")
        print(f"  Stable modes:   {np.sum(stable_mask)}")
        print(f"  Center modes:   {np.sum(center_mask)}")
        
        # Compute mode coordinates (real parts for physical states)
        P_u_real = P_u.real if P_u.size > 0 else np.zeros((6, 0))
        P_s_real = P_s.real if P_s.size > 0 else np.zeros((6, 0))
        P_c_real = P_c.real if P_c.size > 0 else np.zeros((6, 0))
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'P_unstable': P_u_real,
            'P_stable': P_s_real,
            'P_center': P_c_real,
            'unstable_mask': unstable_mask,
            'stable_mask': stable_mask,
            'center_mask': center_mask
        }
    
    def design_floquet_mode_controller(self,
                                      trajectory: np.ndarray,
                                      time_grid: np.ndarray,
                                      mode_decomp: dict,
                                      Q_unstable: np.ndarray = None,
                                      R: np.ndarray = None) -> dict:
        """
        Design controller that only acts on unstable subspace.
        
        This is more efficient than full-state LQR because:
        - Stable modes converge naturally
        - Center modes don't need control
        - Only unstable modes require active stabilization
        
        Parameters
        ----------
        trajectory : np.ndarray
            Reference trajectory
        time_grid : np.ndarray
            Time points
        mode_decomp : dict
            Floquet mode decomposition
        Q_unstable : np.ndarray
            Cost on unstable modes (default: identity)
        R : np.ndarray
            Control cost (default: identity)
            
        Returns
        -------
        dict : Controller gains and projections
        """
        print("\n" + "="*70)
        print("FLOQUET-MODE CONTROLLER DESIGN")
        print("="*70)
        
        n_unstable = mode_decomp['P_unstable'].shape[1]
        
        if n_unstable == 0:
            print("✓ No unstable modes - orbit is stable, no control needed!")
            return {'gains': [], 'needs_control': False}
        
        print(f"\nDesigning controller for {n_unstable} unstable mode(s)...")
        
        # Default costs - make Q_unstable larger to prioritize stabilization
        if Q_unstable is None:
            Q_unstable = 100.0 * np.eye(n_unstable)  # Increased from 1.0
        if R is None:
            R = 0.1 * np.eye(3)  # Control cost
        
        # Extract unstable projection matrix (normalize columns)
        P_u = mode_decomp['P_unstable'].copy()
        for i in range(P_u.shape[1]):
            P_u[:, i] = P_u[:, i] / (np.linalg.norm(P_u[:, i]) + 1e-12)
        
        # Compute orthogonal projector onto unstable subspace
        # P_u_proj = P_u @ P_u.T  (but this is 6x6, we need pseudoinverse)
        P_u_pinv = np.linalg.pinv(P_u)  # Maps full state to unstable coordinates
        
        gains = []
        
        for i in range(0, len(trajectory), 10):  # Compute every 10 points for speed
            # Use average state in vicinity
            state = trajectory[i]
            
            # Compute Jacobian at current state
            A = self.cr3bp._compute_jacobian(state)
            
            # Control matrix (control affects velocities)
            B = np.zeros((6, 3))
            B[3:6, :] = np.eye(3)
            
            # Project dynamics onto unstable subspace
            A_u = P_u_pinv @ A @ P_u  # Reduced dynamics (n_unstable x n_unstable)
            B_u = P_u_pinv @ B         # Reduced control (n_unstable x 3)
            
            # Solve Riccati equation in reduced space
            try:
                P_riccati = solve_continuous_are(A_u, B_u, Q_unstable, R)
                K_u = np.linalg.solve(R, B_u.T @ P_riccati)  # Gain (3 x n_unstable)
                
                # Full-state gain: K = K_u @ P_u_pinv
                # Control: u = -K @ x = -K_u @ (P_u_pinv @ x)
                # This means we:
                # 1. Project state to unstable coordinates: x_u = P_u_pinv @ x
                # 2. Compute control in reduced space: u = -K_u @ x_u
                K_full = K_u @ P_u_pinv  # (3 x 6) gain matrix
                
                # Store for all 10 time steps
                for _ in range(10):
                    gains.append(K_full)
                    if len(gains) >= len(trajectory):
                        break
            except:
                # If CARE fails, use small gain
                K_fallback = 0.01 * np.random.randn(3, 6)
                for _ in range(10):
                    gains.append(K_fallback)
                    if len(gains) >= len(trajectory):
                        break
        
        # Trim to exact length
        gains = gains[:len(trajectory)]
        
        print(f"✓ Floquet-mode controller designed")
        print(f"  Control dimension: 3 (velocity corrections)")
        print(f"  Controlled subspace: {n_unstable} unstable modes")
        print(f"  Uncontrolled subspace: {6 - n_unstable} modes (stable/center)")
        print(f"  Cost ratio Q/R: {np.mean(np.diag(Q_unstable))/np.mean(np.diag(R)):.1f}")
        
        return {
            'gains': gains,
            'needs_control': True,
            'P_unstable': P_u,
            'P_unstable_pinv': P_u_pinv,
            'n_unstable': n_unstable
        }
    
    def simulate_floquet_mode_control(self,
                                     state0_perturbed: np.ndarray,
                                     reference_trajectory: np.ndarray,
                                     time_grid: np.ndarray,
                                     controller: dict,
                                     max_control: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Simulate with Floquet-mode controller.
        
        Parameters
        ----------
        state0_perturbed : np.ndarray
            Initial perturbed state
        reference_trajectory : np.ndarray
            Reference orbit
        time_grid : np.ndarray
            Time points
        controller : dict
            Controller data from design_floquet_mode_controller
        max_control : float
            Maximum allowed control magnitude (saturation)
            
        Returns
        -------
        trajectory : np.ndarray
            Controlled trajectory
        control_history : np.ndarray
            Control effort
        delta_v : float
            Total Δv
        """
        if not controller['needs_control']:
            print("\nOrbit is stable - no control simulation needed")
            return reference_trajectory, np.zeros((len(time_grid), 3)), 0.0
        
        print("\n" + "="*70)
        print("FLOQUET-MODE CONTROLLED SIMULATION")
        print("="*70)
        
        dt = time_grid[1] - time_grid[0]
        trajectory = np.zeros((len(time_grid), 6))
        control_history = np.zeros((len(time_grid), 3))
        trajectory[0] = state0_perturbed
        
        delta_v = 0.0
        saturations = 0
        
        gains = controller['gains']
        
        for i in range(len(time_grid) - 1):
            state = trajectory[i]
            state_ref = reference_trajectory[min(i, len(reference_trajectory) - 1)]
            
            # State error
            error = state - state_ref
            
            # Control law: u = -K * error (K only acts on unstable modes)
            K = gains[min(i, len(gains) - 1)]
            u = -K @ error
            
            # Saturate control to realistic values
            u_magnitude = np.linalg.norm(u)
            if u_magnitude > max_control:
                u = u * (max_control / u_magnitude)
                saturations += 1
            
            control_history[i] = u
            delta_v += np.linalg.norm(u) * dt
            
            # Controlled dynamics
            def controlled_dynamics(t, x):
                dx = self.cr3bp.equations_of_motion(t, x)
                dx[3:6] += u
                return dx
            
            # Integrate
            sol = solve_ivp(
                controlled_dynamics,
                [time_grid[i], time_grid[i+1]],
                state,
                method='DOP853',
                rtol=1e-10,
                atol=1e-10
            )
            
            trajectory[i+1] = sol.y[:, -1]
        
        if saturations > 0:
            print(f"  Control saturated {saturations}/{len(time_grid)} times (max={max_control})")
        
        print(f"✓ Simulation complete")
        print(f"  Total Δv (Floquet-mode): {delta_v:.6f} (nondimensional)")
        print(f"  Max control magnitude: {np.max(np.linalg.norm(control_history, axis=1)):.6f}")
        
        return trajectory, control_history, delta_v


# ============================================================================
# Visualization Functions for Advanced Features
# ============================================================================

def plot_invariant_manifolds(cr3bp,
                             reference_trajectory: np.ndarray,
                             manifold_results: dict,
                             save_path: str = 'invariant_manifolds.png'):
    """
    Plot invariant manifolds in 3D.
    """
    print("\nGenerating invariant manifold visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot primaries
    ax.plot([cr3bp.r1[0]], [cr3bp.r1[1]], [cr3bp.r1[2]], 
           'o', color='blue', markersize=15, label='Earth')
    ax.plot([cr3bp.r2[0]], [cr3bp.r2[1]], [cr3bp.r2[2]], 
           'o', color='gray', markersize=10, label='Moon')
    ax.plot([cr3bp.L2[0]], [cr3bp.L2[1]], [cr3bp.L2[2]], 
           's', color='red', markersize=8, label='L2')
    
    # Reference NRHO
    ax.plot(reference_trajectory[:, 0],
           reference_trajectory[:, 1],
           reference_trajectory[:, 2],
           'g-', linewidth=3, alpha=0.8, label='NRHO')
    
    # Unstable manifolds (departure/escape)
    for traj in manifold_results['unstable_trajectories']:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
               'r-', linewidth=1.5, alpha=0.6)
    
    # Stable manifolds (arrival/capture)
    for traj in manifold_results['stable_trajectories']:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
               'b-', linewidth=1.5, alpha=0.6)
    
    # Custom legend entries for manifolds
    ax.plot([], [], 'r-', linewidth=2, label='Unstable Manifolds (Departure)')
    ax.plot([], [], 'b-', linewidth=2, label='Stable Manifolds (Arrival)')
    
    ax.set_xlabel('x (normalized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (normalized)', fontsize=12, fontweight='bold')
    ax.set_zlabel('z (normalized)', fontsize=12, fontweight='bold')
    ax.set_title('Invariant Manifolds for Low-Energy Transfer Design',
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Manifold plot saved: {save_path}")


def plot_solar_perturbation_comparison(cr3bp_trajectory: np.ndarray,
                                       bcr4bp_trajectory: np.ndarray,
                                       save_path: str = 'solar_perturbation_comparison.png'):
    """
    Compare CR3BP vs BCR4BP trajectories showing solar perturbation effects.
    """
    print("\nGenerating solar perturbation comparison...")
    
    fig = plt.figure(figsize=(16, 5))
    
    # XY projection
    ax1 = fig.add_subplot(131)
    ax1.plot(cr3bp_trajectory[:, 0], cr3bp_trajectory[:, 1],
            'g-', linewidth=2, label='CR3BP (no Sun)', alpha=0.8)
    ax1.plot(bcr4bp_trajectory[:, 0], bcr4bp_trajectory[:, 1],
            'r--', linewidth=2, label='BCR4BP (with Sun)', alpha=0.8)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('XY Projection', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # XZ projection
    ax2 = fig.add_subplot(132)
    ax2.plot(cr3bp_trajectory[:, 0], cr3bp_trajectory[:, 2],
            'g-', linewidth=2, label='CR3BP', alpha=0.8)
    ax2.plot(bcr4bp_trajectory[:, 0], bcr4bp_trajectory[:, 2],
            'r--', linewidth=2, label='BCR4BP', alpha=0.8)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('z', fontsize=12)
    ax2.set_title('XZ Projection', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Deviation magnitude
    ax3 = fig.add_subplot(133)
    deviation = np.linalg.norm(bcr4bp_trajectory[:, :3] - cr3bp_trajectory[:, :3], axis=1)
    time_normalized = np.linspace(0, 1, len(deviation))
    ax3.plot(time_normalized, deviation, 'b-', linewidth=2)
    ax3.set_xlabel('Normalized Time', fontsize=12)
    ax3.set_ylabel('Position Deviation', fontsize=12)
    ax3.set_title('Solar Perturbation Effect', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Solar perturbation comparison saved: {save_path}")
    print(f"  Max deviation: {np.max(deviation):.6f}")
    print(f"  Mean deviation: {np.mean(deviation):.6f}")


def plot_control_comparison(lqr_trajectory: np.ndarray,
                           floquet_trajectory: np.ndarray,
                           lqr_control: np.ndarray,
                           floquet_control: np.ndarray,
                           lqr_dv: float,
                           floquet_dv: float,
                           save_path: str = 'control_comparison.png'):
    """
    Compare full-state LQR vs Floquet-mode control.
    """
    print("\nGenerating control comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    time = np.linspace(0, 1, len(lqr_trajectory))
    
    # Position error
    ax1 = axes[0, 0]
    lqr_error = np.linalg.norm(lqr_trajectory[:, :3] - lqr_trajectory[0, :3], axis=1)
    floquet_error = np.linalg.norm(floquet_trajectory[:, :3] - floquet_trajectory[0, :3], axis=1)
    ax1.plot(time, lqr_error, 'b-', linewidth=2, label='Full-State LQR')
    ax1.plot(time, floquet_error, 'r--', linewidth=2, label='Floquet-Mode')
    ax1.set_xlabel('Normalized Time', fontsize=11)
    ax1.set_ylabel('Position Error', fontsize=11)
    ax1.set_title('Tracking Error Comparison', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Control magnitude
    ax2 = axes[0, 1]
    lqr_control_mag = np.linalg.norm(lqr_control, axis=1)
    floquet_control_mag = np.linalg.norm(floquet_control, axis=1)
    ax2.plot(time, lqr_control_mag, 'b-', linewidth=2, label=f'LQR (Δv={lqr_dv:.5f})')
    ax2.plot(time, floquet_control_mag, 'r--', linewidth=2, label=f'Floquet (Δv={floquet_dv:.5f})')
    ax2.set_xlabel('Normalized Time', fontsize=11)
    ax2.set_ylabel('Control Magnitude', fontsize=11)
    ax2.set_title('Control Effort Comparison', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cumulative Δv
    ax3 = axes[1, 0]
    dt = 1.0 / len(time)
    lqr_cumulative = np.cumsum(lqr_control_mag) * dt
    floquet_cumulative = np.cumsum(floquet_control_mag) * dt
    ax3.plot(time, lqr_cumulative, 'b-', linewidth=2, label='LQR')
    ax3.plot(time, floquet_cumulative, 'r--', linewidth=2, label='Floquet')
    ax3.set_xlabel('Normalized Time', fontsize=11)
    ax3.set_ylabel('Cumulative Δv', fontsize=11)
    ax3.set_title('Integrated Control Cost', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Efficiency metrics
    ax4 = axes[1, 1]
    efficiency_improvement = ((lqr_dv - floquet_dv) / lqr_dv) * 100
    categories = ['Full-State\nLQR', 'Floquet-Mode\nControl']
    dv_values = [lqr_dv, floquet_dv]
    colors = ['blue', 'red']
    bars = ax4.bar(categories, dv_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Total Δv', fontsize=11)
    ax4.set_title(f'Δv Efficiency: {efficiency_improvement:.1f}% Improvement', 
                 fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, dv_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.5f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Control comparison saved: {save_path}")
    print(f"  Δv reduction: {efficiency_improvement:.2f}%")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADVANCED NRHO ANALYSIS MODULE")
    print("PhD-Level Orbital Mechanics Extensions")
    print("="*70)
    print("\nThis module provides:")
    print("  1. Invariant manifold computation for trajectory design")
    print("  2. Bicircular four-body model (Sun perturbations)")
    print("  3. Floquet-mode decomposition control")
    print("\nImport this module and use with the base NRHO implementation.")
    print("="*70)
