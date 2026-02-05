
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import PchipInterpolator
import os
import sys

# Ensure the working directory is this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# sys.path.insert(0, script_dir)
import NSMatterEoS as nsEoS
import ResolverTOV as tov

# ==========================================
# Constants and Units
# ==========================================
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

# ==========================================
# Computation Functions
# ==========================================

def compute_eos_and_mr(params):
    # Rango de densidades para la interpolacion
    dens_max = 1e17 * 1e3 * (1e-45 / m_nuc_MKS)
    dens_min = 5e9 * 1e3 * (1e-45 / m_nuc_MKS)
    n_range = np.logspace(np.log10(dens_min), np.log10(dens_max), 300)
    
    # Check for crust file
    crust_file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'EoS_tables', 'EoS_crust.txt'))
    
    # Compute EoS
    rho_P, pres, ener, dens_sirve, *_ = nsEoS.EoS(n_range, params, add_crust=True, crust_file_path=crust_file_path)
    dens_lim = ener[0]
    
    # Interpolators
    P_rho = PchipInterpolator(ener, pres)
    energia_densidad = PchipInterpolator(dens_sirve, ener)
    
    # Mass-Radius integration
    rhos_central = np.logspace(13.5, 15.6, 200) # Increased range/resolution
    
    masses = []
    radios = []
    
    for rho_m in rhos_central:
        n_bar = rho_m * 1e3 / m_nuc_MKS * 1e-45
        rho0_dim = energia_densidad(n_bar)
        
        R = 1.0 / rho0_dim
        rho_P_pr = lambda P: R * rho_P(P / R)
        P_central_pr = R * P_rho(1 / R)
        dens_lim_pr = R * dens_lim
        rho_nat_to_MKS = 1.0 / MeV_to_fm11 * MeVfm_to_Jm
        
        sol = tov.integrador(rf=50, dr=1e-4, 
                            rho0=rho0_dim * m_nuc**4 / 2 * rho_nat_to_MKS,
                            rho_P=rho_P_pr, P_central=P_central_pr,
                            densidad_limite=dens_lim_pr)
        r_phys, m_phys, *_ = sol
        radios.append(r_phys * 1e-3)
        masses.append(m_phys / 1.989e30)
        
    # Convert EoS for plotting (MeV/fm^3)
    # Filter useful range
    mask = (dens_sirve >= 0.01) & (dens_sirve <= 1.5)
    pres_plot = pres[mask]
    ener_plot = ener[mask]
    
    # Create smooth line
    pres_smooth = np.linspace(pres_plot.min(), pres_plot.max(), 300)
    ener_smooth = rho_P(pres_smooth)
    
    ener_MeV = ener_smooth * (m_nuc**4/2) / MeV_to_fm11
    pres_MeV = pres_smooth * (m_nuc**4/2) / MeV_to_fm11
    
    return ener_MeV, pres_MeV, np.array(radios), np.array(masses)

# ==========================================
# Plotting
# ==========================================

def plot_final_results():
    # Base parameters
    params = [12.684*m_nuc**2, 7.148*m_nuc**2, (4.410*m_nuc**2), 5.610e-3, -6.986e-3]
    
    print("Computing EoS and M-R for base parameters...")
    ener, pres, radios, masses = compute_eos_and_mr(params)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # --- PANEL 1: EoS ---
    ax1.plot(ener, pres, 'k-', linewidth=2.5)
    ax1.set_xlabel(r'Densidad de energía (MeV/fm$^3$)', fontsize=14)
    ax1.set_ylabel(r'Presión (MeV/fm$^3$)', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Causal limit P=rho
    limits = ax1.axis()
    x_causal = np.logspace(np.log10(limits[0]), np.log10(limits[1]), 100)
    ax1.plot(x_causal, x_causal, 'r--', linewidth=1.5, alpha=0.7)
    ax1.text(0.65, 0.85, r'$P = \rho$', transform=ax1.transAxes, fontsize=12, color='red', rotation=28)
    
    # --- PANEL 2: M-R Relation ---
    
    # 1. Horizontal Shaded Regions (Mass only)
    mass_constraints = [
        ('LIGO 2020 (GW190814)', 2.59, 0.09, 'orchid'),
        ('Romani et al 2022 (J0952)', 2.35, 0.17, 'red'),
        ('Fonseca et al 2021 (J0740)', 2.08, 0.07, 'green'),
        ('Antoniadis et al 2013 (J0348)', 2.01, 0.04, 'blue'),
    ]
    
    for label, m, err, color in mass_constraints:
        ax2.axhspan(m - err, m + err, color=color, alpha=0.15)
        # Add label on the right axis or legend? 
        # Using a proxy artist for legend is cleaner given many regions
    
    # 2. Rectangular Shaded Regions (M-R)
    # Format: (Label, M, M_err_up, M_err_down, R, R_err_up, R_err_down, Color)
    mr_constraints = [
        ('Biswas et al 2021 (GW190814)', 2.59, 0.09, 0.09, 14.1, 1.5, 2.0, 'purple'),
        ('Miller et al 2021 (J0740)', 2.08, 0.07, 0.07, 13.7, 2.6, 1.5, 'orange'),
        ('Riley et al 2021 (J0740)', 2.072, 0.067, 0.066, 12.39, 1.30, 0.98, 'cyan'),
        ('Miller et al 2019 (J0030)', 1.44, 0.15, 0.14, 13.02, 1.24, 1.06, 'magenta'),
        ('Riley et al 2019 (J0030)', 1.34, 0.15, 0.16, 12.71, 1.14, 1.19, 'gold'),
        ('Choudhury et al 2024 (J0437)', 1.418, 0.037, 0.037, 11.36, 0.95, 0.63, 'brown'),
        ('Miller et al 2021 (M 1.4)', 1.4, 0.01, 0.01, 12.45, 0.65, 0.65, 'gray') # Tiny mass error for fixed mass constraint
    ]

    # Plot Rectangles
    for label, m, m_up, m_down, r, r_up, r_down, color in mr_constraints:
        width = r_up + r_down
        height = m_up + m_down
        xy = (r - r_down, m - m_down)
        rect = Rectangle(xy, width, height, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.2, label=label)
        ax2.add_patch(rect)
        # Add center point
        ax2.errorbar(r, m, xerr=[[r_down], [r_up]], yerr=[[m_down], [m_up]], fmt='none', ecolor=color, alpha=0.5)

    # Plot Model Curve
    ax2.plot(radios, masses, 'k-', linewidth=2.5, zorder=10)
    
    # Add dummy artists for the horizontal bands to the legend
    for label, m, err, color in mass_constraints:
        ax2.fill_between([], [], color=color, alpha=0.15, label=label)

    ax2.set_xlabel('Radio (km)', fontsize=14)
    ax2.set_ylabel(r'Masa (M$_\odot$)', fontsize=14)
    ax2.set_ylim(-0.45, 3.0)
    ax2.set_xlim(9, 17) # Typical range focus
    
    ax2.legend(fontsize=9, loc='lower left', ncol=2, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('MasaRadio_Final_Thesis.png', dpi=300)
    print("Plot saved to MasaRadio_Final_Thesis.png")
    plt.show()

if __name__ == "__main__":
    plot_final_results()
