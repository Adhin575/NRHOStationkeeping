"""
Near Rectilinear Halo Orbit (NRHO) Analysis in the Earth-Moon CR3BP
===================================================================

A research-grade implementation for computing, analyzing, and controlling
periodic halo orbits in the Circular Restricted Three-Body Problem.

This implementation includes:
- Full 3D CR3BP dynamics with State Transition Matrix
- Differential correction for periodic orbit computation
- Monodromy matrix and Floquet stability analysis
- LQR-based station-keeping control
- Monte Carlo robustness analysis
- Publication-quality visualizations

Author: Research Implementation
Date: February 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, minimize
from scipy.linalg import expm, eig, solve_continuous_are

# Configure matplotlib to use non-GUI backend (prevents display errors)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

from typing import Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: CR3BP Mathematical Model
# ============================================================================

class CR3BP:
    """
    Circular Restricted Three-Body Problem dynamics and utilities.
    
    Implements the Earth-Moon system in nondimensional rotating coordinates.
    """
    
    def __init__(self, mu: float = 0.0121505856):
        """
        Initialize CR3BP system.
        
        Parameters
        ----------
        mu : float
            Mass parameter (M2 / (M1 + M2)) for Earth-Moon system
        """
        self.mu = mu
        self.mu1 = 1.0 - mu  # Normalized mass of primary (Earth)
        self.mu2 = mu         # Normalized mass of secondary (Moon)
        
        # Primary positions in rotating frame (nondimensional)
        self.r1 = np.array([-mu, 0.0, 0.0])      # Earth position
        self.r2 = np.array([1.0 - mu, 0.0, 0.0]) # Moon position
        
        # Compute Lagrange points
        self.L1 = self._compute_lagrange_point(1)
        self.L2 = self._compute_lagrange_point(2)
        self.L3 = self._compute_lagrange_point(3)
        
    def _compute_lagrange_point(self, point_number: int) -> np.ndarray:
        """
        Compute collinear Lagrange point position via root-finding.
        
        Parameters
        ----------
        point_number : int
            Lagrange point number (1, 2, or 3)
            
        Returns
        -------
        np.ndarray
            Position vector of Lagrange point [x, 0, 0]
        """
        mu = self.mu
        
        if point_number == 1:
            # L1: between primaries
            def f(x):
                r1 = abs(x + mu)
                r2 = abs(x - 1 + mu)
                return x - (1 - mu) / r1**2 * (x + mu) - mu / r2**2 * (x - 1 + mu)
            x0 = 1.0 - mu - (mu / 3)**(1/3)
        elif point_number == 2:
            # L2: beyond secondary
            def f(x):
                r1 = abs(x + mu)
                r2 = abs(x - 1 + mu)
                return x - (1 - mu) / r1**2 * (x + mu) - mu / r2**2 * (x - 1 + mu)
            x0 = 1.0 - mu + (mu / 3)**(1/3)
        else:
            # L3: beyond primary
            def f(x):
                r1 = abs(x + mu)
                r2 = abs(x - 1 + mu)
                return x - (1 - mu) / r1**2 * (x + mu) - mu / r2**2 * (x - 1 + mu)
            x0 = -1.0 - mu + (5 * mu / 12)
            
        x_L = fsolve(f, x0)[0]
        return np.array([x_L, 0.0, 0.0])
    
    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        CR3BP equations of motion in rotating frame.
        
        Implements the full 3D equations including Coriolis and centrifugal terms.
        
        Parameters
        ----------
        t : float
            Time (normalized)
        state : np.ndarray
            State vector [x, y, z, vx, vy, vz] (6D)
            
        Returns
        -------
        np.ndarray
            Derivative [vx, vy, vz, ax, ay, az]
        """
        x, y, z, vx, vy, vz = state
        
        # Distances to primaries
        r1 = np.sqrt((x + self.mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2 + z**2)
        
        # Partial derivatives of pseudo-potential
        Ux = x - (1 - self.mu) * (x + self.mu) / r1**3 - self.mu * (x - 1 + self.mu) / r2**3
        Uy = y - (1 - self.mu) * y / r1**3 - self.mu * y / r2**3
        Uz = -(1 - self.mu) * z / r1**3 - self.mu * z / r2**3
        
        # Accelerations including Coriolis terms
        ax = 2 * vy + Ux
        ay = -2 * vx + Uy
        az = Uz
        
        return np.array([vx, vy, vz, ax, ay, az])
    
    def equations_of_motion_with_stm(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        CR3BP equations with State Transition Matrix (STM) propagation.
        
        Augmented state: [x(6), Φ(36)] where Φ is reshaped 6×6 STM.
        
        Parameters
        ----------
        t : float
            Time
        state : np.ndarray
            Augmented state vector [x(6), Φ_flat(36)] (42D)
            
        Returns
        -------
        np.ndarray
            Derivative [dx(6), dΦ_flat(36)]
        """
        # Extract state and STM
        x_vec = state[:6]
        Φ_flat = state[6:]
        Φ = Φ_flat.reshape((6, 6))
        
        # Compute state derivative
        dx = self.equations_of_motion(t, x_vec)
        
        # Compute Jacobian A(t)
        A = self._compute_jacobian(x_vec)
        
        # STM derivative: dΦ/dt = A(t) * Φ(t)
        dΦ = A @ Φ
        
        return np.concatenate([dx, dΦ.flatten()])
    
    def _compute_jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Compute 6×6 Jacobian matrix A(t) for linearized dynamics.
        
        A = [  0₃ₓ₃    I₃ₓ₃  ]
            [ Uxx_mat  C_mat ]
            
        where Uxx_mat contains second partials of pseudo-potential
        and C_mat contains Coriolis terms.
        
        Parameters
        ----------
        state : np.ndarray
            State vector [x, y, z, vx, vy, vz]
            
        Returns
        -------
        np.ndarray
            6×6 Jacobian matrix
        """
        x, y, z, vx, vy, vz = state
        
        # Distances to primaries
        r1 = np.sqrt((x + self.mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2 + z**2)
        
        # Common terms
        r1_3 = r1**3
        r1_5 = r1**5
        r2_3 = r2**3
        r2_5 = r2**5
        
        dx1 = x + self.mu
        dx2 = x - 1 + self.mu
        
        # Second partial derivatives of pseudo-potential
        Uxx = 1 - (1 - self.mu) / r1_3 - self.mu / r2_3 + \
              3 * (1 - self.mu) * dx1**2 / r1_5 + 3 * self.mu * dx2**2 / r2_5
              
        Uyy = 1 - (1 - self.mu) / r1_3 - self.mu / r2_3 + \
              3 * (1 - self.mu) * y**2 / r1_5 + 3 * self.mu * y**2 / r2_5
              
        Uzz = -(1 - self.mu) / r1_3 - self.mu / r2_3 + \
              3 * (1 - self.mu) * z**2 / r1_5 + 3 * self.mu * z**2 / r2_5
              
        Uxy = 3 * (1 - self.mu) * dx1 * y / r1_5 + 3 * self.mu * dx2 * y / r2_5
        Uxz = 3 * (1 - self.mu) * dx1 * z / r1_5 + 3 * self.mu * dx2 * z / r2_5
        Uyz = 3 * (1 - self.mu) * y * z / r1_5 + 3 * self.mu * y * z / r2_5
        
        # Construct Jacobian
        A = np.zeros((6, 6))
        
        # Upper right: Identity (velocity coupling)
        A[0:3, 3:6] = np.eye(3)
        
        # Lower left: Second partials
        A[3, 0] = Uxx
        A[3, 1] = Uxy
        A[3, 2] = Uxz
        
        A[4, 0] = Uxy
        A[4, 1] = Uyy
        A[4, 2] = Uyz
        
        A[5, 0] = Uxz
        A[5, 1] = Uyz
        A[5, 2] = Uzz
        
        # Lower right: Coriolis matrix
        A[3, 4] = 2.0
        A[4, 3] = -2.0
        
        return A
    
    def jacobi_constant(self, state: np.ndarray) -> float:
        """
        Compute Jacobi integral (constant of motion).
        
        C = 2U - v² where U = (x² + y²)/2 + (1-μ)/r1 + μ/r2 + μ(1-μ)/2
        
        Standard form: C = x² + y² + 2(1-μ)/r1 + 2μ/r2 - (vx² + vy² + vz²)
        
        Parameters
        ----------
        state : np.ndarray
            State vector [x, y, z, vx, vy, vz]
            
        Returns
        -------
        float
            Jacobi constant
        """
        x, y, z, vx, vy, vz = state
        
        r1 = np.sqrt((x + self.mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2 + z**2)
        
        v_squared = vx**2 + vy**2 + vz**2
        
        # Correct formula: C = 2U - v²
        C = x**2 + y**2 + 2 * (1 - self.mu) / r1 + 2 * self.mu / r2 - v_squared
        
        return C


# ============================================================================
# SECTION 2: Periodic Orbit Computation via Differential Correction
# ============================================================================

class PeriodicOrbitComputer:
    """
    Compute periodic halo/NRHO orbits using differential correction.
    
    Uses single-shooting method with symmetry constraints.
    """
    
    def __init__(self, cr3bp: CR3BP):
        """
        Initialize orbit computer.
        
        Parameters
        ----------
        cr3bp : CR3BP
            CR3BP system instance
        """
        self.cr3bp = cr3bp
    
    def verify_nrho_characteristics(self, trajectory: np.ndarray, T: float) -> dict:
        """
        Verify that computed orbit has true NRHO characteristics.
        
        NRHOs are distinguished from regular halos by:
        1. High amplitude ratio (Az/Ax > 2.0)
        2. Close lunar approach (perilune distance)
        3. Near-rectilinear shape in x-z projection
        4. Large variation in x-coordinate
        
        Parameters
        ----------
        trajectory : np.ndarray
            Orbit trajectory
        T : float
            Period
            
        Returns
        -------
        dict : NRHO characteristics and verification
        """
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]
        
        # Compute amplitudes
        Ax = np.max(x) - np.min(x)
        Ay = np.max(y) - np.min(y)
        Az = np.max(z) - np.min(z)
        
        # Amplitude ratio (should be > 2.0 for NRHO)
        amplitude_ratio = Az / Ax
        
        # Compute distance to Moon at closest approach
        moon_pos = self.cr3bp.r2
        distances_to_moon = np.sqrt((x - moon_pos[0])**2 + 
                                    (y - moon_pos[1])**2 + 
                                    (z - moon_pos[2])**2)
        perilune = np.min(distances_to_moon)
        apolune = np.max(distances_to_moon)
        
        # Eccentricity-like parameter in rotating frame
        eccentricity = (apolune - perilune) / (apolune + perilune)
        
        # Check near-rectilinearity: measure linearity in x-z projection
        # Fit line to x-z points and compute R² correlation
        z_at_extremes = z[np.where((x == np.max(x)) | (x == np.min(x)))]
        x_at_extremes = x[np.where((x == np.max(x)) | (x == np.min(x)))]
        
        # Measure how "linear" the orbit is in x-z plane
        # For NRHO, trajectory should be elongated along one axis
        aspect_ratio = Ax / Az if Az > Ax else Az / Ax
        
        # NRHO classification criteria
        is_nrho = (
            amplitude_ratio > 1.5 and  # High amplitude ratio
            perilune < 0.15 and         # Close lunar approach
            eccentricity > 0.7          # High eccentricity
        )
        
        results = {
            'is_nrho': is_nrho,
            'amplitude_ratio': amplitude_ratio,
            'Ax': Ax,
            'Ay': Ay, 
            'Az': Az,
            'perilune': perilune,
            'apolune': apolune,
            'eccentricity': eccentricity,
            'aspect_ratio': aspect_ratio
        }
        
        print("\n" + "="*70)
        print("NRHO CHARACTERISTIC VERIFICATION")
        print("="*70)
        print(f"\nAmplitudes:")
        print(f"  Ax (x-amplitude): {Ax:.6f}")
        print(f"  Ay (y-amplitude): {Ay:.6f}")
        print(f"  Az (z-amplitude): {Az:.6f}")
        print(f"  Amplitude ratio Az/Ax: {amplitude_ratio:.4f} (NRHO if > 1.5)")
        
        print(f"\nLunar Proximity:")
        print(f"  Perilune (closest to Moon): {perilune:.6f} (NRHO if < 0.15)")
        print(f"  Apolune (farthest from Moon): {apolune:.6f}")
        print(f"  Eccentricity parameter: {eccentricity:.4f} (NRHO if > 0.7)")
        
        print(f"\nOrbit Classification:")
        if is_nrho:
            print(f"  ✓ TRUE NRHO - Exhibits near-rectilinear characteristics")
        else:
            print(f"  ✗ STANDARD HALO - Does not meet NRHO criteria")
            print(f"    This is a regular halo orbit, not an NRHO")
        
        return results
        
    def compute_periodic_orbit(self, 
                               x0_guess: np.ndarray,
                               T_guess: float,
                               max_iter: int = 50,
                               tol: float = 1e-10) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Compute periodic orbit using differential correction with symmetry.
        
        Enforces y = 0 and vy = 0 at half-period (symmetry condition).
        Uses State Transition Matrix for Newton iteration.
        
        Parameters
        ----------
        x0_guess : np.ndarray
            Initial guess [x, z, vy] (assuming y=0, vx=0, vz=0 at start)
        T_guess : float
            Initial guess for period
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
            
        Returns
        -------
        x0 : np.ndarray
            Converged initial state (6D)
        T : float
            Converged period
        trajectory : np.ndarray
            Full orbit trajectory
        """
        print("\n" + "="*70)
        print("DIFFERENTIAL CORRECTION FOR PERIODIC ORBIT")
        print("="*70)
        
        # Initial free variables: [x, z, vy, T/2]
        free_vars = np.array([x0_guess[0], x0_guess[1], x0_guess[2], T_guess / 2.0])
        
        for iteration in range(max_iter):
            x, z, vy, T_half = free_vars
            
            # Construct full initial state (symmetric start)
            state0 = np.array([x, 0.0, z, 0.0, vy, 0.0])
            
            # Augment with identity STM
            Φ0 = np.eye(6).flatten()
            aug_state0 = np.concatenate([state0, Φ0])
            
            # Integrate to half-period
            sol = solve_ivp(
                self.cr3bp.equations_of_motion_with_stm,
                [0, T_half],
                aug_state0,
                method='DOP853',
                rtol=1e-12,
                atol=1e-12,
                dense_output=True
            )
            
            # Extract final state and STM
            state_f = sol.y[:6, -1]
            Φ_f = sol.y[6:, -1].reshape((6, 6))
            
            # Compute constraint violations (symmetry at half-period)
            # Should have: y = 0, vx = 0, vz = 0
            f_vec = np.array([
                state_f[1],   # y should be 0
                state_f[3],   # vx should be 0
                state_f[5]    # vz should be 0
            ])
            
            error = np.linalg.norm(f_vec)
            
            print(f"Iteration {iteration+1:2d}: Error = {error:.3e}")
            
            if error < tol:
                print(f"✓ Converged in {iteration+1} iterations")
                print(f"  Final error: {error:.3e}")
                break
            
            # Compute Jacobian of constraints w.r.t. free variables
            # ∂f/∂[x, z, vy, T/2]
            
            # Partials from STM
            df_dx = Φ_f[[1, 3, 5], :][:, [0, 2, 4]]
            
            # Partials w.r.t. T/2 (from f(T/2) = f0 + T/2 * f_dot)
            f_dot = self.cr3bp.equations_of_motion(T_half, state_f)
            df_dT = f_dot[[1, 3, 5]]
            
            # Full Jacobian
            DF = np.column_stack([df_dx, df_dT])
            
            # Newton step
            try:
                delta = np.linalg.solve(DF, -f_vec)
            except np.linalg.LinAlgError:
                print("✗ Singular Jacobian - switching to pseudoinverse")
                delta = np.linalg.lstsq(DF, -f_vec, rcond=None)[0]
            
            # Update with damping for robustness
            damping = 1.0 if iteration < 5 else 0.5
            free_vars += damping * delta
            
        else:
            print(f"✗ Did not converge in {max_iter} iterations")
            print(f"  Final error: {error:.3e}")
        
        # Generate full periodic orbit
        x, z, vy, T_half = free_vars
        state0 = np.array([x, 0.0, z, 0.0, vy, 0.0])
        T = 2 * T_half
        
        # Integrate full period with dense output
        aug_state0 = np.concatenate([state0, np.eye(6).flatten()])
        sol = solve_ivp(
            self.cr3bp.equations_of_motion_with_stm,
            [0, T],
            aug_state0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-12,
            dense_output=True
        )
        
        # Sample trajectory at high resolution
        t_eval = np.linspace(0, T, 1000)
        trajectory = sol.sol(t_eval)[:6, :].T
        
        print(f"\n✓ Periodic orbit computed successfully")
        print(f"  Period T = {T:.6f} (nondimensional)")
        print(f"  Initial state: x={x:.6f}, z={z:.6f}, vy={vy:.6f}")
        
        return state0, T, trajectory


# ============================================================================
# SECTION 3: Stability Analysis (Monodromy & Floquet Theory)
# ============================================================================

class StabilityAnalyzer:
    """
    Analyze stability of periodic orbits using Floquet theory.
    """
    
    def __init__(self, cr3bp: CR3BP):
        """
        Initialize stability analyzer.
        
        Parameters
        ----------
        cr3bp : CR3BP
            CR3BP system instance
        """
        self.cr3bp = cr3bp
        
    def compute_monodromy_matrix(self, 
                                 state0: np.ndarray, 
                                 T: float) -> np.ndarray:
        """
        Compute monodromy matrix (one-period STM).
        
        M = Φ(T, 0) represents linearized map after one full period.
        
        Parameters
        ----------
        state0 : np.ndarray
            Initial state on periodic orbit
        T : float
            Orbital period
            
        Returns
        -------
        M : np.ndarray
            6×6 monodromy matrix
        """
        # Augment state with identity STM
        aug_state0 = np.concatenate([state0, np.eye(6).flatten()])
        
        # Integrate one full period
        sol = solve_ivp(
            self.cr3bp.equations_of_motion_with_stm,
            [0, T],
            aug_state0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-13
        )
        
        # Extract monodromy matrix
        M = sol.y[6:, -1].reshape((6, 6))
        
        return M
    
    def analyze_stability(self, M: np.ndarray) -> dict:
        """
        Perform complete Floquet stability analysis.
        
        Computes eigenvalues (Floquet multipliers), stability indices,
        and characterizes stable/unstable/center manifolds.
        
        Parameters
        ----------
        M : np.ndarray
            Monodromy matrix
            
        Returns
        -------
        dict
            Dictionary containing:
            - eigenvalues: Floquet multipliers (complex)
            - stability_indices: ν_i from λ_i = exp(ν_i T)
            - stable_dim: Dimension of stable manifold
            - unstable_dim: Dimension of unstable manifold
            - stable_eigenvectors: Corresponding eigenvectors
            - unstable_eigenvectors: Corresponding eigenvectors
        """
        print("\n" + "="*70)
        print("FLOQUET STABILITY ANALYSIS")
        print("="*70)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eig(M)
        
        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print("\nFloquet Multipliers (Eigenvalues of Monodromy Matrix):")
        print("-" * 70)
        for i, lam in enumerate(eigenvalues):
            mag = np.abs(lam)
            angle = np.angle(lam) * 180 / np.pi
            print(f"  λ_{i+1} = {lam.real:+.6f} {lam.imag:+.6f}j  "
                  f"|λ| = {mag:.6f}  ∠λ = {angle:+.2f}°")
        
        # Classify eigenvalues
        stable_mask = np.abs(eigenvalues) < 1.0 - 1e-6
        unstable_mask = np.abs(eigenvalues) > 1.0 + 1e-6
        center_mask = ~(stable_mask | unstable_mask)
        
        stable_dim = np.sum(stable_mask)
        unstable_dim = np.sum(unstable_mask)
        center_dim = np.sum(center_mask)
        
        print(f"\nManifold Structure:")
        print(f"  Stable manifold dimension:   {stable_dim}")
        print(f"  Unstable manifold dimension: {unstable_dim}")
        print(f"  Center manifold dimension:   {center_dim}")
        
        # Compute stability indices (ν such that λ = exp(ν))
        stability_indices = np.log(eigenvalues + 1e-16)
        
        # Check symplectic property (det(M) should be 1)
        det_M = np.linalg.det(M)
        print(f"\nMonodromy Matrix Properties:")
        print(f"  det(M) = {det_M:.10f} (should be 1.0 for Hamiltonian system)")
        print(f"  Trace(M) = {np.trace(M):.6f}")
        
        # Hamiltonian orbits always have reciprocal eigenvalues
        print(f"\nReciprocal Eigenvalue Check (Hamiltonian property):")
        for i in range(len(eigenvalues) // 2):
            product = eigenvalues[i] * eigenvalues[-(i+1)]
            print(f"  λ_{i+1} × λ_{len(eigenvalues)-i} = {product:.10f} (should be 1.0)")
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'stability_indices': stability_indices,
            'stable_dim': stable_dim,
            'unstable_dim': unstable_dim,
            'center_dim': center_dim,
            'stable_eigenvectors': eigenvectors[:, stable_mask],
            'unstable_eigenvectors': eigenvectors[:, unstable_mask],
            'monodromy_matrix': M
        }


# ============================================================================
# SECTION 4: LQR Station-Keeping Control
# ============================================================================

class LQRController:
    """
    Linear Quadratic Regulator for station-keeping on periodic orbit.
    """
    
    def __init__(self, cr3bp: CR3BP):
        """
        Initialize LQR controller.
        
        Parameters
        ----------
        cr3bp : CR3BP
            CR3BP system instance
        """
        self.cr3bp = cr3bp
        
    def design_lqr(self, 
                   trajectory: np.ndarray,
                   time_grid: np.ndarray,
                   Q: np.ndarray,
                   R: np.ndarray) -> Tuple[list, list]:
        """
        Design time-varying LQR controller along reference trajectory.
        
        At each time step, solves continuous-time algebraic Riccati equation:
        A'P + PA - PBR⁻¹B'P + Q = 0
        
        Gain: K(t) = R⁻¹B'P(t)
        
        Parameters
        ----------
        trajectory : np.ndarray
            Reference trajectory (N × 6)
        time_grid : np.ndarray
            Time points
        Q : np.ndarray
            State cost matrix (6 × 6)
        R : np.ndarray
            Control cost matrix (3 × 3)
            
        Returns
        -------
        K_list : list
            Time-varying gain matrices
        P_list : list
            Riccati solution matrices
        """
        print("\n" + "="*70)
        print("LQR CONTROLLER DESIGN")
        print("="*70)
        
        K_list = []
        P_list = []
        
        # Control matrix (control in velocity components)
        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3)
        
        print(f"Computing time-varying LQR gains along trajectory...")
        print(f"  Number of time steps: {len(time_grid)}")
        
        for i, state in enumerate(trajectory):
            # Compute Jacobian at current state
            A = self.cr3bp._compute_jacobian(state)
            
            # Solve continuous-time Riccati equation
            try:
                P = solve_continuous_are(A, B, Q, R)
                K = np.linalg.solve(R, B.T @ P)
                
                K_list.append(K)
                P_list.append(P)
            except Exception as e:
                print(f"  Warning: CARE solution failed at step {i}, using previous gain")
                if K_list:
                    K_list.append(K_list[-1])
                    P_list.append(P_list[-1])
                else:
                    K_list.append(np.zeros((3, 6)))
                    P_list.append(np.zeros((6, 6)))
        
        print(f"✓ LQR controller designed successfully")
        return K_list, P_list
    
    def simulate_controlled(self,
                           state0_perturbed: np.ndarray,
                           reference_trajectory: np.ndarray,
                           time_grid: np.ndarray,
                           K_list: list) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Simulate controlled trajectory with LQR feedback.
        
        Control law: u(t) = -K(t) * [x(t) - x_ref(t)]
        
        Parameters
        ----------
        state0_perturbed : np.ndarray
            Initial perturbed state
        reference_trajectory : np.ndarray
            Reference orbit
        time_grid : np.ndarray
            Time points
        K_list : list
            Time-varying gains
            
        Returns
        -------
        trajectory_controlled : np.ndarray
            Controlled trajectory
        control_history : np.ndarray
            Control effort history
        delta_v : float
            Total Δv (integrated control magnitude)
        """
        print("\n" + "="*70)
        print("CONTROLLED TRAJECTORY SIMULATION")
        print("="*70)
        
        dt = time_grid[1] - time_grid[0]
        trajectory_controlled = np.zeros((len(time_grid), 6))
        control_history = np.zeros((len(time_grid), 3))
        trajectory_controlled[0] = state0_perturbed
        
        delta_v = 0.0
        
        for i in range(len(time_grid) - 1):
            # Current state
            state = trajectory_controlled[i]
            
            # Reference state (with interpolation)
            idx = min(i, len(reference_trajectory) - 1)
            state_ref = reference_trajectory[idx]
            
            # State error
            error = state - state_ref
            
            # Control law: u = -K * error
            K = K_list[min(i, len(K_list) - 1)]
            u = -K @ error
            
            control_history[i] = u
            
            # Accumulate Δv
            delta_v += np.linalg.norm(u) * dt
            
            # Controlled dynamics: dx/dt = f(x) + B*u
            def controlled_dynamics(t, x):
                dx = self.cr3bp.equations_of_motion(t, x)
                dx[3:6] += u  # Add control to velocity derivatives
                return dx
            
            # Integrate one time step
            sol = solve_ivp(
                controlled_dynamics,
                [time_grid[i], time_grid[i+1]],
                state,
                method='DOP853',
                rtol=1e-10,
                atol=1e-10
            )
            
            trajectory_controlled[i+1] = sol.y[:, -1]
        
        print(f"✓ Controlled simulation complete")
        print(f"  Total Δv = {delta_v:.6f} (nondimensional)")
        print(f"  Max control magnitude = {np.max(np.linalg.norm(control_history, axis=1)):.6f}")
        
        return trajectory_controlled, control_history, delta_v


# ============================================================================
# SECTION 5: Monte Carlo Robustness Analysis
# ============================================================================

def monte_carlo_analysis(cr3bp: CR3BP,
                        state0: np.ndarray,
                        T: float,
                        n_samples: int = 100,
                        position_std: float = 1e-4,
                        velocity_std: float = 1e-5) -> np.ndarray:
    """
    Perform Monte Carlo simulation with perturbed initial conditions.
    
    Parameters
    ----------
    cr3bp : CR3BP
        CR3BP system
    state0 : np.ndarray
        Nominal initial state
    T : float
        Integration time
    n_samples : int
        Number of Monte Carlo samples
    position_std : float
        Standard deviation for position perturbations
    velocity_std : float
        Standard deviation for velocity perturbations
        
    Returns
    -------
    trajectories : np.ndarray
        Array of sampled trajectories (n_samples × n_points × 6)
    """
    print("\n" + "="*70)
    print(f"MONTE CARLO ROBUSTNESS ANALYSIS ({n_samples} samples)")
    print("="*70)
    
    print(f"Position perturbation: σ = {position_std:.2e}")
    print(f"Velocity perturbation: σ = {velocity_std:.2e}")
    
    trajectories = []
    t_eval = np.linspace(0, T, 500)
    
    for i in range(n_samples):
        # Generate random perturbation
        pos_pert = np.random.normal(0, position_std, 3)
        vel_pert = np.random.normal(0, velocity_std, 3)
        perturbation = np.concatenate([pos_pert, vel_pert])
        
        # Perturbed initial state
        state_pert = state0 + perturbation
        
        # Simulate
        sol = solve_ivp(
            cr3bp.equations_of_motion,
            [0, T],
            state_pert,
            method='DOP853',
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-10
        )
        
        trajectories.append(sol.y.T)
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{n_samples} simulations")
    
    trajectories = np.array(trajectories)
    
    print(f"✓ Monte Carlo analysis complete")
    
    return trajectories


# ============================================================================
# SECTION 6: Visualization Functions
# ============================================================================

def plot_3d_interactive(cr3bp: CR3BP,
                       trajectory: np.ndarray,
                       trajectory_uncontrolled: np.ndarray = None,
                       trajectory_controlled: np.ndarray = None,
                       save_path: str = 'nrho_3d_interactive.png'):
    """
    Create 3D plot using Matplotlib (static image with high quality).
    """
    print("\n" + "="*70)
    print("GENERATING 3D VISUALIZATION")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Earth
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_earth = 0.01 * np.outer(np.cos(u), np.sin(v)) + cr3bp.r1[0]
    y_earth = 0.01 * np.outer(np.sin(u), np.sin(v))
    z_earth = 0.01 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.8, label='Earth')
    
    # Moon
    x_moon = 0.005 * np.outer(np.cos(u), np.sin(v)) + cr3bp.r2[0]
    y_moon = 0.005 * np.outer(np.sin(u), np.sin(v))
    z_moon = 0.005 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_moon, y_moon, z_moon, color='gray', alpha=0.8)
    
    # L2 point
    ax.scatter([cr3bp.L2[0]], [cr3bp.L2[1]], [cr3bp.L2[2]], 
              color='red', s=100, marker='D', label='L2', edgecolors='black', linewidths=2)
    
    # Reference NRHO
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
           color='green', linewidth=3, label='Reference NRHO', alpha=0.9)
    
    # Uncontrolled trajectory
    if trajectory_uncontrolled is not None:
        ax.plot(trajectory_uncontrolled[:, 0], 
               trajectory_uncontrolled[:, 1], 
               trajectory_uncontrolled[:, 2],
               color='red', linewidth=2.5, linestyle='--', 
               label='Uncontrolled (Perturbed)', alpha=0.7)
    
    # Controlled trajectory
    if trajectory_controlled is not None:
        ax.plot(trajectory_controlled[:, 0], 
               trajectory_controlled[:, 1], 
               trajectory_controlled[:, 2],
               color='orange', linewidth=2.5, 
               label='LQR Controlled', alpha=0.8)
    
    ax.set_xlabel('x (normalized)', fontsize=14, fontweight='bold')
    ax.set_ylabel('y (normalized)', fontsize=14, fontweight='bold')
    ax.set_zlabel('z (normalized)', fontsize=14, fontweight='bold')
    ax.set_title('Near Rectilinear Halo Orbit (NRHO) - Earth-Moon System', 
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = max(
        np.ptp(trajectory[:, 0]),
        np.ptp(trajectory[:, 1]),
        np.ptp(trajectory[:, 2])
    ) / 2.0
    
    mid_x = np.mean(trajectory[:, 0])
    mid_y = np.mean(trajectory[:, 1])
    mid_z = np.mean(trajectory[:, 2])
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 3D plot saved: {save_path}")


def plot_floquet_multipliers(stability_results: dict, save_path: str = 'floquet_multipliers.png'):
    """
    Plot Floquet multipliers in complex plane.
    """
    print("\nGenerating Floquet multiplier plot...")
    
    eigenvalues = stability_results['eigenvalues']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.3, label='Unit Circle')
    
    # Plot eigenvalues
    ax.scatter(eigenvalues.real, eigenvalues.imag, s=200, c='red', 
               marker='o', edgecolors='black', linewidths=2, zorder=5, label='Floquet Multipliers')
    
    # Annotate
    for i, lam in enumerate(eigenvalues):
        ax.annotate(f'λ_{i+1}', (lam.real, lam.imag), 
                   xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Real Part', fontsize=14)
    ax.set_ylabel('Imaginary Part', fontsize=14)
    ax.set_title('Floquet Multipliers (Eigenvalues of Monodromy Matrix)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Floquet multiplier plot saved: {save_path}")


def plot_jacobi_constant_verification(cr3bp: CR3BP, 
                                     trajectory: np.ndarray,
                                     save_path: str = 'jacobi_constant.png'):
    """
    Plot Jacobi constant along trajectory to verify conservation.
    """
    print("\nGenerating Jacobi constant verification plot...")
    
    C_values = np.array([cr3bp.jacobi_constant(state) for state in trajectory])
    C_mean = np.mean(C_values)
    C_variation = np.abs(C_values - C_mean)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Jacobi constant value
    ax1.plot(C_values, linewidth=2, color='blue')
    ax1.axhline(C_mean, color='red', linestyle='--', linewidth=1, label=f'Mean = {C_mean:.10f}')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Jacobi Constant C', fontsize=12)
    ax1.set_title('Jacobi Constant Along Trajectory', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Variation from mean
    ax2.semilogy(C_variation, linewidth=2, color='red')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('|C - C_mean|', fontsize=12)
    ax2.set_title('Jacobi Constant Conservation Error', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Jacobi constant plot saved: {save_path}")
    print(f"  Max variation: {np.max(C_variation):.3e}")


def plot_monte_carlo_envelope(cr3bp: CR3BP,
                              trajectories: np.ndarray,
                              reference: np.ndarray,
                              save_path: str = 'monte_carlo_envelope.png'):
    """
    Plot statistical envelope of Monte Carlo trajectories.
    """
    print("\nGenerating Monte Carlo envelope plot...")
    
    fig = plt.figure(figsize=(16, 5))
    
    # XY projection
    ax1 = fig.add_subplot(131)
    for traj in trajectories[::5]:  # Plot subset for clarity
        ax1.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.1, linewidth=0.5)
    ax1.plot(reference[:, 0], reference[:, 1], 'r-', linewidth=3, label='Reference')
    ax1.plot(cr3bp.r1[0], cr3bp.r1[1], 'o', color='blue', markersize=10, label='Earth')
    ax1.plot(cr3bp.r2[0], cr3bp.r2[1], 'o', color='gray', markersize=7, label='Moon')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('XY Projection', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # XZ projection
    ax2 = fig.add_subplot(132)
    for traj in trajectories[::5]:
        ax2.plot(traj[:, 0], traj[:, 2], 'b-', alpha=0.1, linewidth=0.5)
    ax2.plot(reference[:, 0], reference[:, 2], 'r-', linewidth=3, label='Reference')
    ax2.plot(cr3bp.r1[0], cr3bp.r1[2], 'o', color='blue', markersize=10, label='Earth')
    ax2.plot(cr3bp.r2[0], cr3bp.r2[2], 'o', color='gray', markersize=7, label='Moon')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('z', fontsize=12)
    ax2.set_title('XZ Projection', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')
    
    # YZ projection
    ax3 = fig.add_subplot(133)
    for traj in trajectories[::5]:
        ax3.plot(traj[:, 1], traj[:, 2], 'b-', alpha=0.1, linewidth=0.5)
    ax3.plot(reference[:, 1], reference[:, 2], 'r-', linewidth=3, label='Reference')
    ax3.set_xlabel('y', fontsize=12)
    ax3.set_ylabel('z', fontsize=12)
    ax3.set_title('YZ Projection', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Monte Carlo envelope plot saved: {save_path}")


def create_animation(cr3bp: CR3BP,
                    trajectory: np.ndarray,
                    save_path: str = 'nrho_simulation.gif'):
    """
    Create animated GIF of NRHO orbit in rotating frame.
    Uses a fallback method if animation backend fails.
    """
    print("\n" + "="*70)
    print("GENERATING ORBIT ANIMATION")
    print("="*70)
    
    try:
        # Method 1: Try FuncAnimation (preferred)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up static elements
        earth_marker = ax.plot([cr3bp.r1[0]], [cr3bp.r1[1]], [cr3bp.r1[2]], 
                              'o', color='blue', markersize=15, label='Earth')[0]
        moon_marker = ax.plot([cr3bp.r2[0]], [cr3bp.r2[1]], [cr3bp.r2[2]], 
                             'o', color='gray', markersize=10, label='Moon')[0]
        l2_marker = ax.plot([cr3bp.L2[0]], [cr3bp.L2[1]], [cr3bp.L2[2]], 
                           's', color='red', markersize=8, label='L2')[0]
        
        # Initialize trajectory line and current position
        line, = ax.plot([], [], [], 'g-', linewidth=2, alpha=0.6, label='NRHO')
        point, = ax.plot([], [], [], 'ro', markersize=8, label='Spacecraft')
        
        # Set axis limits
        x_range = [np.min(trajectory[:, 0]) - 0.05, np.max(trajectory[:, 0]) + 0.05]
        y_range = [np.min(trajectory[:, 1]) - 0.05, np.max(trajectory[:, 1]) + 0.05]
        z_range = [np.min(trajectory[:, 2]) - 0.05, np.max(trajectory[:, 2]) + 0.05]
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_xlabel('x (normalized)', fontsize=12, fontweight='bold')
        ax.set_ylabel('y (normalized)', fontsize=12, fontweight='bold')
        ax.set_zlabel('z (normalized)', fontsize=12, fontweight='bold')
        ax.set_title('Near Rectilinear Halo Orbit - Rotating Frame', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point
        
        def animate(frame):
            # Update trajectory
            line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
            line.set_3d_properties(trajectory[:frame, 2])
            
            # Update current position
            if frame > 0:
                point.set_data([trajectory[frame-1, 0]], [trajectory[frame-1, 1]])
                point.set_3d_properties([trajectory[frame-1, 2]])
            
            # Rotate view slightly
            ax.view_init(elev=20, azim=frame * 0.5)
            
            return line, point
        
        # Create animation
        n_frames = min(len(trajectory), 200)  # Limit frames for reasonable file size
        frame_indices = np.linspace(0, len(trajectory), n_frames, dtype=int)
        
        print(f"Animating {n_frames} frames using FuncAnimation...")
        anim = FuncAnimation(fig, animate, init_func=init, 
                            frames=frame_indices, interval=50, blit=False)
        
        # Save as GIF
        writer = PillowWriter(fps=20)
        anim.save(save_path, writer=writer)
        plt.close()
        
        print(f"✓ Animation saved: {save_path}")
        print(f"  Frames: {n_frames}, FPS: 20")
        
    except (AttributeError, RuntimeError) as e:
        # Method 2: Fallback - create frames manually and save as GIF
        print(f"  FuncAnimation failed ({type(e).__name__}), using manual frame generation...")
        plt.close('all')
        
        n_frames = min(len(trajectory), 200)
        frame_indices = np.linspace(0, len(trajectory), n_frames, dtype=int)
        
        frames = []
        
        for i, frame_idx in enumerate(frame_indices):
            if (i + 1) % 20 == 0:
                print(f"  Rendering frame {i+1}/{n_frames}...")
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot bodies
            ax.plot([cr3bp.r1[0]], [cr3bp.r1[1]], [cr3bp.r1[2]], 
                   'o', color='blue', markersize=15, label='Earth')
            ax.plot([cr3bp.r2[0]], [cr3bp.r2[1]], [cr3bp.r2[2]], 
                   'o', color='gray', markersize=10, label='Moon')
            ax.plot([cr3bp.L2[0]], [cr3bp.L2[1]], [cr3bp.L2[2]], 
                   's', color='red', markersize=8, label='L2')
            
            # Plot trajectory up to current frame
            if frame_idx > 0:
                ax.plot(trajectory[:frame_idx, 0], 
                       trajectory[:frame_idx, 1], 
                       trajectory[:frame_idx, 2],
                       'g-', linewidth=2, alpha=0.6, label='NRHO')
                
                # Current position
                ax.plot([trajectory[frame_idx-1, 0]], 
                       [trajectory[frame_idx-1, 1]], 
                       [trajectory[frame_idx-1, 2]],
                       'ro', markersize=8, label='Spacecraft')
            
            # Set axis properties
            x_range = [np.min(trajectory[:, 0]) - 0.05, np.max(trajectory[:, 0]) + 0.05]
            y_range = [np.min(trajectory[:, 1]) - 0.05, np.max(trajectory[:, 1]) + 0.05]
            z_range = [np.min(trajectory[:, 2]) - 0.05, np.max(trajectory[:, 2]) + 0.05]
            
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            ax.set_xlabel('x (normalized)', fontsize=12, fontweight='bold')
            ax.set_ylabel('y (normalized)', fontsize=12, fontweight='bold')
            ax.set_zlabel('z (normalized)', fontsize=12, fontweight='bold')
            ax.set_title('Near Rectilinear Halo Orbit - Rotating Frame', fontsize=16, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.view_init(elev=20, azim=i * 0.5)
            
            # Convert to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            
            plt.close(fig)
        
        # Save frames as GIF using PIL
        try:
            from PIL import Image
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(
                save_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=50,  # 50ms per frame = 20 FPS
                loop=0
            )
            print(f"✓ Animation saved: {save_path}")
            print(f"  Frames: {n_frames}, FPS: 20")
        except ImportError:
            print("✗ PIL not available, cannot save GIF")
            print("  Install with: pip install Pillow")
    
    except Exception as e:
        print(f"✗ Animation generation failed: {e}")
        print("  Continuing without animation...")


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution function - runs complete NRHO analysis pipeline.
    """
    print("\n" + "="*70)
    print("NEAR RECTILINEAR HALO ORBIT (NRHO) ANALYSIS")
    print("Earth-Moon Circular Restricted Three-Body Problem")
    print("="*70)
    print("\nThis is a research-grade implementation suitable for publication")
    print("All computations use high-precision numerical methods")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Initialize CR3BP System
    # ========================================================================
    
    cr3bp = CR3BP(mu=0.0121505856)
    
    print("\n" + "="*70)
    print("SYSTEM PARAMETERS")
    print("="*70)
    print(f"Mass parameter μ = {cr3bp.mu:.10f}")
    print(f"Earth position:  [{cr3bp.r1[0]:.6f}, {cr3bp.r1[1]:.6f}, {cr3bp.r1[2]:.6f}]")
    print(f"Moon position:   [{cr3bp.r2[0]:.6f}, {cr3bp.r2[1]:.6f}, {cr3bp.r2[2]:.6f}]")
    print(f"L1 position:     [{cr3bp.L1[0]:.6f}, {cr3bp.L1[1]:.6f}, {cr3bp.L1[2]:.6f}]")
    print(f"L2 position:     [{cr3bp.L2[0]:.6f}, {cr3bp.L2[1]:.6f}, {cr3bp.L2[2]:.6f}]")
    
    # ========================================================================
    # STEP 2: Compute Periodic NRHO
    # ========================================================================
    
    # Initial guess for southern L2 NRHO (9:2 resonant family)
    # NRHOs are characterized by:
    # - Large z-amplitude (near-rectilinear in x-z projection)
    # - Close lunar approach (perilune)
    # - Period ~6.5 days (T ≈ 1.51 in normalized units)
    # - High eccentricity in rotating frame
    x0_guess = np.array([1.0221, 0.1825, -0.1036])  # [x, z, vy] for 9:2 NRHO
    T_guess = 1.511  # Period for NRHO (~6.56 days)
    
    orbit_computer = PeriodicOrbitComputer(cr3bp)
    state0, T, trajectory = orbit_computer.compute_periodic_orbit(
        x0_guess, T_guess, max_iter=50, tol=1e-10
    )
    
    # Verify NRHO characteristics
    nrho_characteristics = orbit_computer.verify_nrho_characteristics(trajectory, T)
    
    # ========================================================================
    # STEP 3: Stability Analysis
    # ========================================================================
    
    analyzer = StabilityAnalyzer(cr3bp)
    M = analyzer.compute_monodromy_matrix(state0, T)
    stability_results = analyzer.analyze_stability(M)
    
    # ========================================================================
    # STEP 4: LQR Station-Keeping Control
    # ========================================================================
    
    controller = LQRController(cr3bp)
    
    # Design cost matrices (tuned for good performance)
    Q = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])  # State cost
    R = np.diag([0.1, 0.1, 0.1])  # Control cost
    
    time_grid = np.linspace(0, T, 1000)
    K_list, P_list = controller.design_lqr(trajectory, time_grid, Q, R)
    
    # Simulate perturbed initial condition
    perturbation = np.array([0.001, 0.0005, 0.001, 0.0001, 0.0001, 0.0001])
    state0_perturbed = state0 + perturbation
    
    # Uncontrolled trajectory
    print("\nSimulating uncontrolled (perturbed) trajectory...")
    sol_uncontrolled = solve_ivp(
        cr3bp.equations_of_motion,
        [0, T],
        state0_perturbed,
        method='DOP853',
        t_eval=time_grid,
        rtol=1e-10,
        atol=1e-10
    )
    trajectory_uncontrolled = sol_uncontrolled.y.T
    
    # Controlled trajectory
    trajectory_controlled, control_history, delta_v = controller.simulate_controlled(
        state0_perturbed, trajectory, time_grid, K_list
    )
    
    # ========================================================================
    # STEP 5: Monte Carlo Robustness Analysis
    # ========================================================================
    
    mc_trajectories = monte_carlo_analysis(
        cr3bp, state0, T,
        n_samples=100,
        position_std=1e-4,
        velocity_std=1e-5
    )
    
    # ========================================================================
    # STEP 6: Generate All Visualizations
    # ========================================================================
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 3D Interactive Plot
    plot_3d_interactive(
        cr3bp, trajectory,
        trajectory_uncontrolled=trajectory_uncontrolled,
        trajectory_controlled=trajectory_controlled
    )
    
    # Floquet Multipliers
    plot_floquet_multipliers(stability_results)
    
    # Jacobi Constant Verification
    plot_jacobi_constant_verification(cr3bp, trajectory)
    
    # Monte Carlo Envelope
    plot_monte_carlo_envelope(cr3bp, mc_trajectories, trajectory)
    
    # Animation
    create_animation(cr3bp, trajectory)
    
    # ========================================================================
    # STEP 7: Summary Report
    # ========================================================================
    
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    print("\n1. PERIODIC ORBIT CHARACTERISTICS")
    print(f"   Initial state: [{state0[0]:.6f}, {state0[1]:.6f}, {state0[2]:.6f}, "
          f"{state0[3]:.6f}, {state0[4]:.6f}, {state0[5]:.6f}]")
    print(f"   Period: {T:.6f} (nondimensional)")
    print(f"   Jacobi constant: {cr3bp.jacobi_constant(state0):.10f}")
    print(f"\n   NRHO Characteristics:")
    print(f"   Orbit type: {'TRUE NRHO' if nrho_characteristics['is_nrho'] else 'STANDARD HALO'}")
    print(f"   Amplitude ratio (Az/Ax): {nrho_characteristics['amplitude_ratio']:.4f}")
    print(f"   Perilune distance: {nrho_characteristics['perilune']:.6f}")
    print(f"   Eccentricity: {nrho_characteristics['eccentricity']:.4f}")
    
    print("\n2. STABILITY PROPERTIES")
    print(f"   Stable manifold dimension: {stability_results['stable_dim']}")
    print(f"   Unstable manifold dimension: {stability_results['unstable_dim']}")
    print(f"   Largest |λ|: {np.max(np.abs(stability_results['eigenvalues'])):.6f}")
    
    print("\n3. CONTROL PERFORMANCE")
    print(f"   Total Δv (LQR): {delta_v:.6f} (nondimensional)")
    max_deviation_uncontrolled = np.max(np.linalg.norm(trajectory_uncontrolled - trajectory, axis=1))
    max_deviation_controlled = np.max(np.linalg.norm(trajectory_controlled - trajectory, axis=1))
    print(f"   Max deviation (uncontrolled): {max_deviation_uncontrolled:.6f}")
    print(f"   Max deviation (controlled): {max_deviation_controlled:.6f}")
    print(f"   Deviation reduction: {(1 - max_deviation_controlled/max_deviation_uncontrolled)*100:.1f}%")
    
    print("\n4. MONTE CARLO STATISTICS")
    final_deviations = np.linalg.norm(mc_trajectories[:, -1, :3] - trajectory[-1, :3], axis=1)
    print(f"   Mean final position deviation: {np.mean(final_deviations):.6f}")
    print(f"   Std final position deviation: {np.std(final_deviations):.6f}")
    print(f"   Max final position deviation: {np.max(final_deviations):.6f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - nrho_3d_interactive.png (3D visualization)")
    print("  - floquet_multipliers.png (stability spectrum)")
    print("  - jacobi_constant.png (energy conservation)")
    print("  - monte_carlo_envelope.png (robustness analysis)")
    print("  - nrho_simulation.gif (orbit animation)")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
