import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import os
import sys

# Ensure the working directory is this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# sys.path.insert(0, script_dir)
import NSMatterEoS as nsEoS
import ResolverTOV as tov

# Physical constants in MKS and conversion factors
hbar_MKS = 1.0545718e-34  # J s
c_MKS = 299792458        # m/s
G_MKS = 6.67430e-11      # m^3/kg/s^2
proton_mass = 1.6726219e-27  # kg
neutron_mass = 1.6749275e-27 # kg
m_nuc_MKS = (proton_mass + neutron_mass) / 2.0  # kg
e_MKS = 1.602186021766e-19  # C
# Conversion factors
Kg_to_fm11 = c_MKS / hbar_MKS * 1e-15
MeV_to_fm11 = e_MKS / (hbar_MKS * c_MKS * 1e9)
MeVfm_to_Jm = 1e51 * e_MKS
# Derived natural units
m_nuc = m_nuc_MKS * Kg_to_fm11

def compute_eos_and_mr(params, n_range=None, rhos_central=None, crust_file_path=None, rf=30.0, dr=3e-5):
    """
    Compute equation of state and mass-radius relation for given parameters.
    
    Parameters
    ----------
    params : list
        List of 5 parameters [A_sigma, A_omega, A_rho, b, c]
    n_range : array, optional
        Baryon density range for EoS computation
    rhos_central : array, optional
        Central mass densities for TOV integration (in kg/m^3)
    crust_file_path : str, optional
        Path to crust EoS file
    rf : float, optional
        Final radius for TOV integration (default: 30.0)
    dr : float, optional
        Step size for TOV integration (default: 3e-5)
        
    Returns
    -------
    dens_sirve : array
        Baryon density (fm^-3)
    rho_P : PchipInterpolator
        Interpolator for energy density as function of pressure
    pres : array
        Pressure (units of m**4/2)
    masses : array
        Stellar masses (M_sun)
    radios : array
        Stellar radii (km)
    compacs : array
        Compactness (GM/c^2R)
    """
    # Default values
    if n_range is None:
        dens_max = 1e17 * 1e3 * (1e-45 / m_nuc_MKS)
        dens_min = 5e9 * 1e3 * (1e-45 / m_nuc_MKS)
        n_range = np.logspace(np.log10(dens_min), np.log10(dens_max), 250)
    
    if rhos_central is None:
        rhos_central = np.logspace(13.5, 15.5, 150)
    
    if crust_file_path is None:
        crust_file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'EoS_tables', 'EoS_crust.txt'))
    
    # Compute EoS
    rho_P, pres, ener, dens_sirve, *_ = nsEoS.EoS(n_range, params, add_crust=True, crust_file_path=crust_file_path)
    dens_lim = ener[0]
    
    # Create interpolators
    P_rho = PchipInterpolator(ener, pres)
    energia_densidad = PchipInterpolator(dens_sirve, ener)
    
    # Compute mass-radius relation
    masses = np.zeros_like(rhos_central)
    radios = np.zeros_like(rhos_central)
    compacs = np.zeros_like(rhos_central)
    
    for k, rho_m in enumerate(rhos_central):
        n_bar = rho_m * 1e3 / m_nuc_MKS * 1e-45
        rho0_dim = energia_densidad(n_bar)
        R = 1.0 / rho0_dim
        rho_P_pr = lambda P: R * rho_P(P / R)
        P_central_pr = R * P_rho(1 / R)
        dens_lim_pr = R * dens_lim
        rho_nat_to_MKS = 1.0 / MeV_to_fm11 * MeVfm_to_Jm
        
        sol = tov.integrador(rf=rf, dr=dr,
                            rho0=rho0_dim * m_nuc**4 / 2 * rho_nat_to_MKS,
                            rho_P=rho_P_pr, P_central=P_central_pr,
                            densidad_limite=dens_lim_pr)
        r_phys, m_phys, *_ = sol
        compacs[k] = G_MKS * m_phys / c_MKS**2 / r_phys
        radios[k] = r_phys * 1e-3
        masses[k] = m_phys / 1.989e30
    
    return dens_sirve, rho_P, pres, masses, radios, compacs

def plot_parameter_variation(base_params, param_index, param_values, param_name=None, 
                             n_range=None, rhos_central=None, colormap=plt.cm.copper_r,
                             figsize=(12, 6), rf=30.0, dr=3e-5, 
                             dens_min=4e-2, dens_max=1.0):
    """
    Plot EoS and M-R relation for different values of a single parameter.
    
    Parameters
    ----------
    base_params : list
        Base parameters [A_sigma, A_omega, A_rho, b, c]
    param_index : int
        Index of parameter to vary (0-4)
    param_values : array-like
        Values of the parameter to explore
    param_name : str, optional
        Name of the parameter for plot labels
    n_range : array, optional
        Baryon density range for EoS computation
    rhos_central : array, optional
        Central mass densities for TOV integration
    colormap : matplotlib colormap, optional
        Colormap for the different parameter values
    figsize : tuple, optional
        Figure size (width, height)
    rf : float, optional
        Final radius for TOV integration (default: 30.0)
    dr : float, optional
        Step size for TOV integration (default: 3e-5)
    dens_min : float, optional
        Minimum baryon density for EoS plot (fm^-3, default: 4e-2)
    dens_max : float, optional
        Maximum baryon density for EoS plot (fm^-3, default: 1.0)
        
    Returns
    -------
    fig : matplotlib figure
        The generated figure
    """
    param_names = [r'$A_\sigma$', r'$A_\omega$', r'$A_\rho$', r'$b$', r'$c$']
    
    if param_name is None:
        param_name = param_names[param_index]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Generate colors and line styles
    colors = colormap(np.linspace(0, 1, len(param_values)))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # solid, dashed, dashdot, dotted, custom
    
    # Store results for finding max mass
    all_masses = []
    all_radios = []
    
    for i, param_val in enumerate(param_values):
        # Create modified parameters
        params = base_params.copy()
        params[param_index] = param_val
        
        # Print current parameter name and value
        print(f"\nComputing for {param_name} = {param_val}\n")
        
        # Compute EoS and M-R
        dens_sirve, rho_P, pres, masses, radios, compacs = compute_eos_and_mr(
            params, n_range=n_range, rhos_central=rhos_central, rf=rf, dr=dr
        )
        
        all_masses.append(masses)
        all_radios.append(radios)
        
        # Filter for baryon density in specified range
        mask_dens = (dens_sirve >= dens_min) & (dens_sirve <= dens_max)
        pres_filtered = pres[mask_dens]
        
        # Create pressure range for interpolation
        pres_min, pres_max = pres_filtered.min(), pres_filtered.max()
        pres_interp = np.linspace(pres_min, pres_max, 300)
        
        # Get energy density from interpolation
        ener_interp = rho_P(pres_interp)
        
        # Convert to physical units for plotting
        # Energy density: natural units to MeV/fm^3
        ener_MeV_fm3 = ener_interp * (m_nuc**4/2) / MeV_to_fm11
        # Pressure: natural units to MeV/fm^3
        pres_MeV_fm3 = pres_interp * (m_nuc**4/2) / MeV_to_fm11
        
        # Select line style
        ls = linestyles[i % len(linestyles)]
        
        # Plot EoS
        ax1.plot(ener_MeV_fm3, pres_MeV_fm3, color=colors[i], linestyle=ls, linewidth=1,
                label=f'{param_name}={param_val:.2f}' if param_index in [2] else 
                      f'{param_name}={param_val:.3e}' if param_index in [3, 4] else
                      f'{param_name}={param_val:.1f}')
        
        # Filter M-R for radius < 20 km
        mask_radius = radios < 20
        radios_filtered = radios[mask_radius]
        masses_filtered = masses[mask_radius]
        
        # Plot M-R relation
        ax2.plot(radios_filtered, masses_filtered, color=colors[i], linestyle=ls, linewidth=1)
    
    # Format EoS plot
    ax1.set_xlabel(r'Densidad de energía (MeV/fm$^3$)', fontsize=14)
    ax1.set_ylabel(r'Presión (MeV/fm$^3$)', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=14, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Get x and y limits from data (min and max across all curves)
    all_lines = ax1.get_lines()
    if all_lines:
        all_xdata = np.concatenate([line.get_xdata() for line in all_lines])
        all_ydata = np.concatenate([line.get_ydata() for line in all_lines])
        x_min, x_max = np.min(all_xdata), np.max(all_xdata)
        y_min = np.min(all_ydata)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(bottom=y_min, top = x_max)
    
    # Add causal limit (P = rho) using the configured x limits
    x_min, x_max = ax1.get_xlim()
    causal_x = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    ax1.plot(causal_x, causal_x, 'r-', alpha=0.5, linewidth=1.5)
    
    # Add text annotation for causal limit
    ax1.text(0.7, 0.95, r'$P = \rho$', transform=ax1.transAxes, fontsize=14,
             rotation=20, verticalalignment='top')
    
    # Format M-R plot
    ax2.set_xlabel('Radio (km)', fontsize=14)
    ax2.set_ylabel(r'Masa (M$_\odot$)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Adjust M-R plot limits: tight on sides and bottom, margin on top
    ax2.autoscale(enable=True, axis='both', tight=True)
    # ax2.margins(x=0.02, y=0)  # Small margin on x, none on y
    ylims = ax2.get_ylim()
    y_range = ylims[1] - ylims[0]
    ax2.set_ylim(ylims[0], ylims[1] + 0.05 * y_range)  # Add 5% margin on top
    
    plt.tight_layout()
    
    return fig

def plot_multiple_parameters(base_params, variations, n_range=None, rhos_central=None,
                             figsize=(12, 12), colormap=plt.cm.copper_r, rf=30.0, dr=3e-5,
                             dens_min=4e-2, dens_max=1.0):
    """
    Plot EoS and M-R for variations in multiple parameters in a grid layout.
    
    Parameters
    ----------
    base_params : list
        Base parameters [A_sigma, A_omega, A_rho, b, c]
    variations : dict
        Dictionary with keys as parameter indices and values as arrays of parameter values
        Example: {0: np.linspace(190, 260, 5), 2: np.linspace(4, 6, 5)}
    n_range : array, optional
        Baryon density range for EoS computation
    rhos_central : array, optional
        Central mass densities for TOV integration
    figsize : tuple, optional
        Figure size (width, height)
    colormap : matplotlib colormap, optional
        Colormap for the different parameter values
    rf : float, optional
        Final radius for TOV integration (default: 30.0)
    dr : float, optional
        Step size for TOV integration (default: 3e-5)
    dens_min : float, optional
        Minimum baryon density for EoS plot (fm^-3, default: 4e-2)
    dens_max : float, optional
        Maximum baryon density for EoS plot (fm^-3, default: 1.0)
        
    Returns
    -------
    fig : matplotlib figure
        The generated figure
    """
    param_names = [r'$A_\sigma$', r'$A_\omega$', r'$A_\rho$', r'$b$', r'$c$']
    
    n_variations = len(variations)
    fig, axes = plt.subplots(n_variations, 2, figsize=figsize)
    
    # Ensure axes is 2D
    if n_variations == 1:
        axes = axes.reshape(1, -1)
    
    for row, (param_index, param_values) in enumerate(variations.items()):
        ax1, ax2 = axes[row, 0], axes[row, 1]
        param_name = param_names[param_index]
        
        # Generate colors and line styles
        colors = colormap(np.linspace(0, 1, len(param_values)))
        linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
        
        for i, param_val in enumerate(param_values):
            # Create modified parameters
            params = base_params.copy()
            params[param_index] = param_val
            
            # Compute EoS and M-R
            dens_sirve, rho_P, pres, masses, radios, compacs = compute_eos_and_mr(
                params, n_range=n_range, rhos_central=rhos_central, rf=rf, dr=dr
            )
            
            # Filter for baryon density in specified range
            mask_dens = (dens_sirve >= dens_min) & (dens_sirve <= dens_max)
            pres_filtered = pres[mask_dens]
            
            # Create pressure range for interpolation
            pres_min, pres_max = pres_filtered.min(), pres_filtered.max()
            pres_interp = np.linspace(pres_min, pres_max, 300)
            
            # Get energy density from interpolation
            ener_interp = rho_P(pres_interp)
            
            # Convert to physical units
            ener_MeV_fm3 = ener_interp * (m_nuc**4/2) / MeV_to_fm11
            pres_MeV_fm3 = pres_interp * (m_nuc**4/2) / MeV_to_fm11
            
            # Select line style
            ls = linestyles[i % len(linestyles)]
            
            # Plot EoS
            ax1.plot(ener_MeV_fm3, pres_MeV_fm3, color=colors[i], linestyle=ls, linewidth=1,
                    label=f'{param_name}={param_val:.2f}' if param_index in [2] else 
                          f'{param_name}={param_val:.3e}' if param_index in [3, 4] else
                          f'{param_name}={param_val:.1f}')
            
            # Filter M-R for radius < 20 km
            mask_radius = radios < 20
            radios_filtered = radios[mask_radius]
            masses_filtered = masses[mask_radius]
            
            # Plot M-R relation
            ax2.plot(radios_filtered, masses_filtered, color=colors[i], linestyle=ls, linewidth=1)
        
        # Format EoS plot
        ax1.set_xlabel(r'Densidad de energía (MeV/fm$^3$)', fontsize=14)
        ax1.set_ylabel(r'Presión (MeV/fm$^3$)', fontsize=14)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(fontsize=14, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Get x and y limits from data (min and max across all curves)
        all_lines = ax1.get_lines()
        if all_lines:
            all_xdata = np.concatenate([line.get_xdata() for line in all_lines])
            all_ydata = np.concatenate([line.get_ydata() for line in all_lines])
            x_min, x_max = np.min(all_xdata), np.max(all_xdata)
            y_min = np.min(all_ydata)
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(bottom=y_min)
        
        # Add causal limit (P = rho) using the configured x limits
        x_min, x_max = ax1.get_xlim()
        causal_x = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        ax1.plot(causal_x, causal_x, 'r-', alpha=0.5, linewidth=1.5)
        
        # Add text annotation for causal limit
        ax1.text(0.7, 0.95, r'$P = \rho$', transform=ax1.transAxes, fontsize=14,
                 rotation=20, verticalalignment='top')
        
    # Format M-R plot
    ax2.set_xlabel('Radio (km)', fontsize=14)
    ax2.set_ylabel(r'Masa (M$_\odot$)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Adjust M-R plot limits: tight on sides and bottom, margin on top
    ax2.autoscale(enable=True, axis='both', tight=True)
    ax2.margins(x=0.02, y=0)  # Small margin on x, none on y
    ylims = ax2.get_ylim()
    y_range = ylims[1] - ylims[0]
    ax2.set_ylim(ylims[0], ylims[1] + 0.05 * y_range)  # Add 5% margin on top
    
    plt.tight_layout()
    
    return fig

# Example usage
if __name__ == "__main__":
    from time import time
    
    # Base parameters: [A_sigma, A_omega, A_rho, b, c]
    base_params = [12.684*m_nuc**2, 7.148*m_nuc**2, (4.410*m_nuc**2), 5.610e-3, -6.986e-3]
    
    # Rango de densidades para la interpolacion
    dens_max = 1e17 * 1e3 * (1e-45 / m_nuc_MKS)
    dens_min = 5e9 * 1e3 * (1e-45 / m_nuc_MKS)
    n_range = np.logspace(np.log10(dens_min), np.log10(dens_max), 250)
    
    # Rango de densidades de masa para TOV
    rhos_central = np.logspace(13.5, 15.5, 150)
    
    # Parametros de integración
    rf = 30.0
    dr = 2e-5
    # dr = 1e-4
    # dr = 1e-3
    
    # Densidad mínima y máxima para graficar EoS
    dens_min_plot = 4e-2
    # dens_min_plot = 2.5e-2
    dens_max_plot = 1.0
    # dens_max_plot = 1.5
    
    print("Base parameters:")
    param_names = ['A_sigma', 'A_omega', 'A_rho', 'b', 'c']
    for name, val in zip(param_names, base_params):
        print(f"  {name} = {val:.6e}")
    
    # Example 1: Vary One Parameter
    print("\n=== Example 1: Varying One Parameter ===")
    t0 = time()
    # A_sigma_values = np.linspace(12.0*m_nuc**2, 13.5*m_nuc**2, 5)
    param_values = np.linspace(-7e-3, 7e-3, 5)
    index = 4
    fig1 = plot_parameter_variation(base_params, param_index=index, param_values=param_values,
                                    n_range=n_range, rhos_central=rhos_central,
                                    dens_min=dens_min_plot, dens_max=dens_max_plot,
                                    rf=rf, dr=dr)
    t1 = time()
    print(f"Time: {t1-t0:.2f} s")
    
    # # Example 2: Vary A_rho
    # print("\n=== Example 2: Varying A_rho ===")
    # t0 = time()
    # A_rho_values = np.linspace(3.5*m_nuc**2, 5.5*m_nuc**2, 5)
    # fig2 = plot_parameter_variation(base_params, param_index=2, param_values=A_rho_values)
    # t1 = time()
    # print(f"Time: {t1-t0:.2f} s")
    
    # # Example 3: Multiple parameters
    # print("\n=== Example 3: Multiple parameters ===")
    # t0 = time()
    # variations = {
    #     0: np.linspace(12.0*m_nuc**2, 13.5*m_nuc**2, 4),  # A_sigma
    #     2: np.linspace(3.5*m_nuc**2, 5.5*m_nuc**2, 4),    # A_rho
    # }
    # fig3 = plot_multiple_parameters(base_params, variations, figsize=(16, 8))
    # t1 = time()
    # print(f"Time: {t1-t0:.2f} s")
    
    plt.show()
