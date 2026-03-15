Autonomous Low-Thrust Station-Keeping in Earth-Moon NRHO
A research-grade Python implementation for computing, analyzing, and controlling spacecraft in Near Rectilinear Halo Orbits (NRHOs) within the Earth-Moon Circular Restricted Three-Body Problem (CR3BP). This codebase supports the full pipeline from orbit computation and stability analysis through to sensor-fusion navigation and optimal station-keeping control.

Overview
Near Rectilinear Halo Orbits are the baseline orbit for NASA's Lunar Gateway. Their near-rectilinear geometry provides near-continuous Earth visibility and favorable lunar south-pole coverage, but their natural instability means active station-keeping is required throughout a mission's lifetime. This project implements the core astrodynamics, estimation, and control algorithms needed to study that problem at a graduate research level.

Repository Structure
.
├── nrho_complete_analysis.py       # Base CR3BP model, orbit computation, stability, LQR control
├── NRHO_advanced_analysis.py       # Advanced pipeline: manifolds, BCR4BP, Floquet control
├── Visualization.py                # Animated EKF sensor-fusion navigation demo
└── Autonomous_Low-Thrust_Station-Keeping_in_Earth_Moon_NRHO.pptx

Modules
nrho_complete_analysis.py — Base Implementation
The foundation of the codebase. Implements the full 3D CR3BP dynamics and all core analysis tools.
Key classes and functions:

CR3BP — Earth-Moon CR3BP dynamics in a nondimensional rotating frame. Computes Lagrange points, equations of motion (including Coriolis and centrifugal terms), State Transition Matrix propagation, and the Jacobi constant.
PeriodicOrbitComputer — Differential correction algorithm for computing periodic NRHO solutions. Identifies true NRHOs by their amplitude ratio and perilune distance.
StabilityAnalyzer — Monodromy matrix construction and Floquet stability analysis. Classifies stable, unstable, and center-subspace eigenvalues.
LQRController — Full-state Linear Quadratic Regulator for reference trajectory tracking. Integrates Δv accumulation for fuel budget estimation.
monte_carlo_analysis — Statistical robustness analysis over ensembles of perturbed initial conditions.
Visualization utilities: plot_3d_interactive, plot_floquet_multipliers, plot_jacobi_constant_verification, plot_monte_carlo_envelope, create_animation.


NRHO_advanced_analysis.py — Advanced Analysis Pipeline
Extends the base module with higher-fidelity dynamics and more sophisticated control. Imports all public classes and functions from nrho_complete_analysis.py.
Sections:
A — Invariant Manifold Analysis (InvariantManifoldAnalyzer)
Computes and propagates the stable and unstable manifolds of the NRHO by decomposing the monodromy matrix eigenstructure. Supports trajectory design applications including low-energy Earth-Moon transfers, ballistic lunar capture, and Gateway insertion trajectories.
B — Bicircular Restricted Four-Body Problem (BCR4BPPerturbation)
Models solar gravity as a time-varying perturbation on top of the CR3BP. Quantifies the tidal acceleration magnitude (typically 1×10⁻⁷ to 1×10⁻⁵ nondimensional units) and tracks how single-orbit and multi-orbit deviations grow over time.
C — Floquet Mode Control (FloquetModeController)
Optimal control strategy that acts only in the unstable subspace of the orbit, rather than on the full state vector. Reduces unnecessary actuation in stable/center directions. Δv performance is compared directly against the full-state LQR from the base module.
run_complete_advanced_analysis() — Top-level driver that orchestrates all of the above, prints a structured mission-design summary (including physics-based annual Δv estimates benchmarked against published Gateway values of 5–20 m/s/year), and saves three publication-ready figures:

advanced_invariant_manifolds.png
advanced_solar_perturbations.png
advanced_control_comparison.png


Visualization.py — Sensor Fusion Navigation Demo
An animated, standalone script demonstrating real-time navigation using an Extended Kalman Filter (EKF) that fuses two complementary sensor modalities:

GNSS — range measurement to a reference point; low-frequency, absolute position information.
Optical — line-of-sight unit vector to the Moon; high-frequency, attitude-referenced bearing.

The EKF propagates the CR3BP dynamics as its process model, linearizes measurement functions numerically at each step, and applies sequential GNSS then optical updates every timestep. The resulting animation overlays the true trajectory against the filter estimate, illustrating how fusion of the two sensors rapidly corrects initial state error.

Physical System
ParameterValueSystemEarth-Moon CR3BPMass parameter μ0.0121505856Characteristic length384,400 kmCharacteristic time~4.343 daysVelocity unit~1,023.2 m/sNominal NRHO period~6.5 days (~55 orbits/year)Perilune altitude~3,000 km above lunar surfaceAnnual Δv budget (literature)5–20 m/s/year
All computations are performed in nondimensional rotating-frame coordinates unless otherwise noted. Physical units are quoted in summary outputs for mission-design context.

Dependencies
numpy
scipy
matplotlib
Install with:
bashpip install numpy scipy matplotlib

Usage
Run the base analysis:
bashpython nrho_complete_analysis.py
Run the full advanced pipeline (requires nrho_complete_analysis.py in the same directory or on PYTHONPATH):
bashpython NRHO_advanced_analysis.py
Launch the animated navigation demo:
bashpython Visualization.py

Outputs
FileDescriptionadvanced_invariant_manifolds.pngStable/unstable manifold tubes in the rotating frameadvanced_solar_perturbations.pngSolar tidal perturbation magnitude and orbit deviation growthadvanced_control_comparison.pngLQR vs. Floquet-mode Δv and trajectory deviationMatplotlib animation (live)True vs. EKF-estimated trajectory (Visualization.py)

Key Results

Orbit stability: The L2 southern NRHO has one unstable eigenvalue pair with magnitude significantly greater than unity, requiring active control. All other modes are center or marginally stable.
Solar perturbations: The BCR4BP model shows secular deviation growth across multiple orbits, confirming that single-orbit corrections are insufficient for long-duration missions.
Control comparison: Floquet-mode control targets only the unstable subspace, potentially reducing unnecessary Δv expenditure relative to full-state LQR. Performance is sensitive to Q/R tuning.
Navigation: The EKF sensor-fusion filter converges quickly from a 1–2% initial state error when combining GNSS range and optical line-of-sight measurements.
