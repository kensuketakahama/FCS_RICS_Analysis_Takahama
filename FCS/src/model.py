import numpy as np

def fcs_standard_model(tau, N, D, w0, wz, T, tau_trip, y0):
    """
    Standard FCS model
    G(tau) = (1/N) * Diff(tau) * Trip(tau) + y0
    """
    N = max(N, 1e-3)
    D = max(D, 1e-9)
    w0 = max(w0, 1e-3)
    wz = max(wz, 1e-3)
    
    tau_D = (w0**2) / (4.0 * D)
    S = wz / w0
    
    # Diffusion
    diff_rad = 1.0 + (tau / tau_D)
    diff_ax  = 1.0 + (tau / ((S**2) * tau_D))
    diff_term = (diff_rad)**(-1) * (diff_ax)**(-0.5)
    
    # Triplet
    if 0 < T < 1.0:
        trip_term = 1.0 + (T / (1.0 - T)) * np.exp(-tau / max(tau_trip, 1e-9))
    else:
        trip_term = 1.0
        
    return (1.0 / N) * diff_term * trip_term + y0