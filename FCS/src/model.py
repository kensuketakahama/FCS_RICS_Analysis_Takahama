import numpy as np

def rics_3d_equation(xy, D, G0, w0, wz, pixel_size, pixel_dwell, line_time, 
                     T=0.0, tau_trip=0.0, use_triplet=False):
    """
    RICS 3D diffusion model.
    Optional: Triplet blinking term.
    """
    xi = xy[0] * pixel_size  # spatial lag x (m)
    psi = xy[1] * pixel_size # spatial lag y (m)
    
    # Time lag at (xi, psi)
    tau = np.abs(xi * pixel_dwell / pixel_size + psi * line_time / pixel_size)
    
    # Diffusion Term
    tau_p = (w0**2) / (4 * D)
    term1 = (1 + (4 * D * tau) / (w0**2)) ** -1
    term2 = (1 + (4 * D * tau) / (wz**2)) ** -0.5
    G_diff = term1 * term2
    
    # Triplet Term
    G_trip = 1.0
    if use_triplet and T > 0 and tau_trip > 0:
        # Avoid division by zero or log errors
        if T >= 1.0: T = 0.999
        G_trip = 1.0 + (T / (1.0 - T)) * np.exp(-tau / tau_trip)
    
    # Scan Spatial Correlation (S_scan is approximated as exp term in some derivations, 
    # but standard RICS usually incorporates scan geometry into tau. 
    # Here we use the standard form: G = G0 * G_diff * G_trip * exp(...))
    # Note: The simple form used in typical RICS software assumes the spatial Gaussian beam overlap 
    # is handled by the tau definition for raster scanning.
    # We apply the exp factor for beam displacement separate from diffusion.
    
    # Spatial exponential term for scanning
    arg_exp = -((xi**2 + psi**2) / (w0**2)) * ((1 + (4*D*tau)/(w0**2))**-1) # Corrections for diffusion broadening during scan
    # A simplified RICS often separates the spatial overlap from diffusion dynamics:
    # G(xi, psi) = G0 * G_diff(tau) * G_trip(tau) * exp(...)
    
    # Standard RICS equation (Digman et al.)
    # G(xi, psi) = S(xi, psi) * G_diff(xi, psi)
    # S(xi, psi) = exp( - ( (xi^2 + psi^2) / w0^2 ) ) <-- This is for pure spatial, but usually coupled with diffusion
    
    # Let's use the explicit coupled form:
    # w0^2 -> w0^2 + 4D*tau
    
    w0_sq_t = w0**2 + 4 * D * tau
    wz_sq_t = wz**2 + 4 * D * tau
    
    # Correct 3D Gaussian overlap integral with diffusion
    G_val = G0 * (w0**2 / w0_sq_t) * (wz / np.sqrt(wz_sq_t)) * np.exp( - (xi**2 + psi**2) / w0_sq_t )
    
    if use_triplet:
        G_val *= G_trip
        
    return G_val.ravel()