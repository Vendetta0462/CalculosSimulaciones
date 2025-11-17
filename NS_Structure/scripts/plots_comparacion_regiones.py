"""
Script para comparar ecuaciones de estado y relaciones masa-radio
entre diferentes regiones del espacio de parámetros.

Compara los parámetros que producen la mayor masa en diferentes
archivos de malla estelar.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import NSMatterEoS as nsEoS
import ResolverTOV as tov

# Usamos el directorio del archivo
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Constantes físicas en MKS
hbar_MKS = 1.0545718e-34  # J s
c_MKS = 299792458  # m/s
G_MKS = 6.67430e-11  # m^3/kg/s^2
proton_mass = 1.6726219e-27  # kg
neutron_mass = 1.6749275e-27  # kg
m_nuc_MKS = (proton_mass + neutron_mass) / 2.0  # kg
e_MKS = 1.602186021766e-19  # C

# Conversiones útiles
Kg_to_fm11 = c_MKS / hbar_MKS * 1e-15  # kg to fm^-1
MeV_to_fm11 = e_MKS / (hbar_MKS * c_MKS * 1e9)  # MeV to fm^-1
MeVfm_to_Jm = 1e51 * e_MKS  # MeV/fm to J/m

# Constantes en unidades naturales
m_nuc = m_nuc_MKS * Kg_to_fm11  # fm^-1


def find_max_mass_params(filepath):
    """
    Encuentra los parámetros que producen la masa máxima en un archivo de malla.
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo .npz con los datos de la malla
        
    Returns:
    --------
    params : list
        Lista con los 5 parámetros [A_sigma, A_omega, A_rho, b, c]
    max_mass : float
        Valor de la masa máxima encontrada
    """
    data = np.load(filepath)
    
    A_sigma_range = data['A_sigma_range']
    A_omega_range = data['A_omega_range']
    params_base = data['params']
    mass_mesh = data['mass_mesh']
    
    # Encontrar el índice de la masa máxima (ignorando NaNs)
    max_idx = np.nanargmax(mass_mesh)
    i, j = np.unravel_index(max_idx, mass_mesh.shape)
    
    max_mass = mass_mesh[i, j]
    A_sigma_max = A_sigma_range[j]
    A_omega_max = A_omega_range[i]
    
    # Construir lista completa de parámetros
    params = [A_sigma_max, A_omega_max, params_base[2], params_base[3], params_base[4]]
    
    return params, max_mass


def compute_eos_and_mr(params, rf=30.0, dr=3e-5):
    """
    Calcula la ecuación de estado y la relación masa-radio para un conjunto de parámetros.
    
    Parameters:
    -----------
    params : list
        Lista con los 5 parámetros [A_sigma, A_omega, A_rho, b, c]
    rf : float
        Radio final para la integración de TOV (en unidades de c=G=M_sun=1)
    dr : float
        Paso de integración para TOV
        
    Returns:
    --------
    dens_sirve : array
        Densidades bariónicas útiles
    rho_P : callable
        Interpolador de densidad de energía en función de presión
    pres : array
        Presiones
    masses : array
        Masas en masas solares
    radios : array
        Radios en km
    compacs : array
        Compacidades
    """
    # Rango de densidades para la EoS
    densidad_masa_max = 1e17 * 1e3 * (1e-45 / m_nuc_MKS)
    densidad_masa_min = 5e9 * 1e3 * (1e-45 / m_nuc_MKS)
    n_barions = np.logspace(np.log10(densidad_masa_min), np.log10(densidad_masa_max), 200)
    
    # Calcular EoS con corteza
    rho_P, pres, ener, dens_sirve, _ = nsEoS.EoS(
        n_barions, params,
        add_crust=True,
        crust_file_path='../EoS_tables/EoS_crust.txt'
    )
    
    # Preparar para integración TOV
    dens_lim = ener[0]
    P_rho = PchipInterpolator(ener, pres)
    densidad_energia = PchipInterpolator(dens_sirve, ener)
    
    # Rango de densidades centrales
    rhos_central = np.logspace(13.5, 15.5, 150)
    masses = np.zeros_like(rhos_central)
    radios = np.zeros_like(rhos_central)
    compacs = np.zeros_like(rhos_central)
    
    # Integrar TOV para cada densidad central
    rho_nat_to_MKS = 1.0 / MeV_to_fm11 * MeVfm_to_Jm
    
    for i, rho_m in enumerate(rhos_central):
        n_bar = rho_m * 1e3 / m_nuc_MKS * 1e-45
        
        # Calcular densidad de energía adimensional
        rho0_dim = densidad_energia(n_bar)
        
        # Adimensionalizar con rho0
        R = 1.0 / rho0_dim
        rho_P_prima = lambda P: R * rho_P(P / R)
        P_central_prima = R * P_rho(1 / R)
        dens_lim_prima = R * dens_lim
        
        # Integrar TOV
        sol = tov.integrador(
            rf=rf, dr=dr,
            rho0=rho0_dim * m_nuc**4 / 2 * rho_nat_to_MKS,
            rho_P=rho_P_prima,
            P_central=P_central_prima,
            densidad_limite=dens_lim_prima
        )
        
        r_phys, m_phys = sol[0], sol[1]
        radios[i] = r_phys * 1e-3  # km
        masses[i] = m_phys / 1.989e30  # M_sun
        compacs[i] = G_MKS * m_phys / c_MKS**2 / r_phys
    
    return dens_sirve, rho_P, pres, masses, radios, compacs


def plot_comparison(files_dict, dens_min=4e-2, dens_max=1.0, rf=30.0, dr=3e-5):
    """
    Compara las EoS y relaciones M-R de diferentes regiones del espacio de parámetros.
    
    Parameters:
    -----------
    files_dict : dict
        Diccionario con etiquetas como claves y rutas de archivos como valores
    dens_min : float
        Densidad bariónica mínima para graficar EoS (fm^-3)
    dens_max : float
        Densidad bariónica máxima para graficar EoS (fm^-3)
    rf : float
        Radio final adimensional para integración TOV
    dr : float
        Paso de integración TOV
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Colores y estilos de línea
    colors = plt.cm.copper_r(np.linspace(0.2, 0.9, len(files_dict)))
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    # Factor de conversión a MeV/fm^3
    conversion_factor = m_nuc**4 / 2 / MeV_to_fm11
    
    # Para ajustar límites de los ejes
    all_P_vals = []
    all_rho_vals = []
    all_R_vals = []
    all_M_vals = []
    
    # Procesar cada archivo
    for idx, (label, filepath) in enumerate(files_dict.items()):
        print(f"\nProcesando {label}...\n")
        
        # Encontrar parámetros de masa máxima
        params, _ = find_max_mass_params(filepath)
        
        # Calcular EoS y M-R
        dens_sirve, rho_P, pres, masses, radios, compacs = compute_eos_and_mr(params, rf, dr)
        
        # Filtrar densidades para EoS
        mask_dens = (dens_sirve >= dens_min) & (dens_sirve <= dens_max)
        pres_filtered = pres[mask_dens]
        
        # Interpolar para obtener curvas suaves
        P_vals = np.logspace(np.log10(pres_filtered[0]), np.log10(pres_filtered[-1]), 300)
        rho_vals = rho_P(P_vals)
        
        all_P_vals.extend(P_vals)
        all_rho_vals.extend(rho_vals)
        
        # Filtrar radios < 20 km
        mask_r = radios < 20
        radios_filtered = radios[mask_r]
        masses_filtered = masses[mask_r]
        
        all_R_vals.extend(radios_filtered)
        all_M_vals.extend(masses_filtered)
        
        # Crear etiqueta con parámetros (solo parámetros en una línea)
        param_label = (f"$A_\\sigma$={params[0]:.1f}, $A_\\omega$={params[1]:.1f}, "
                      f"$A_\\rho$={params[2]:.1f},\n$b$={params[3]:.2e}, $c$={params[4]:.2e}")
        
        # Convertir a MeV/fm^3
        P_vals_MeV = P_vals * conversion_factor
        rho_vals_MeV = rho_vals * conversion_factor
        
        # Graficar EoS (P en y, rho en x)
        color = colors[idx]
        ls = line_styles[idx % len(line_styles)]
        axes[0].loglog(rho_vals_MeV, P_vals_MeV, color=color, linestyle=ls, linewidth=1, label=param_label)
        
        # Graficar M-R
        axes[1].plot(radios_filtered, masses_filtered, color=color, linestyle=ls, linewidth=1)
        
        print(f"  Masa máxima: {np.max(masses):.3f} M_sun")
        print(f"  Parámetros: A_σ={params[0]:.1f}, A_ω={params[1]:.1f}, A_ρ={params[2]:.1f}, b={params[3]:.2e}, c={params[4]:.2e}")
    
    # Configurar subplot de EoS
    axes[0].set_xlabel(r'Densidad de energía (MeV/fm$^3$)', fontsize=14)
    axes[0].set_ylabel(r'Presión (MeV/fm$^3$)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Get x and y limits from data (min and max across all curves)
    all_lines = axes[0].get_lines()
    if all_lines:
        all_xdata = np.concatenate([line.get_xdata() for line in all_lines])
        all_ydata = np.concatenate([line.get_ydata() for line in all_lines])
        x_min, x_max = np.min(all_xdata), np.max(all_xdata)
        y_min = np.min(all_ydata)
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_ylim(bottom=y_min, top=x_max)
    
    # Límite causal P = ρ (sin label) using configured x limits
    x_min, x_max = axes[0].get_xlim()
    causal_x = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    axes[0].plot(causal_x, causal_x, 'r-', linewidth=1, zorder=10)
    
    # Add text annotation for causal limit
    axes[0].text(0.7, 0.95, r'$P = \rho$', transform=axes[0].transAxes, fontsize=14,
                 rotation=20, verticalalignment='top')
    
    # Configurar subplot de M-R
    axes[1].set_xlabel(r'Radio (km)', fontsize=14)
    axes[1].set_ylabel(r'Masa ($M_\odot$)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Adjust M-R plot limits: tight on sides and bottom, margin on top
    axes[1].autoscale(enable=True, axis='both', tight=True)
    ylims = axes[1].get_ylim()
    y_range = ylims[1] - ylims[0]
    axes[1].set_ylim(ylims[0], ylims[1] + 0.05 * y_range)  # Add 5% margin on top
    
    # Leyendas
    axes[0].legend(fontsize=10, loc='lower right', labelspacing=1.2)
    # axes[1].legend(fontsize=12, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    from time import time
    
    # Definir archivos a comparar
    base_path = '../results/EspacioParametros/'
    files_to_compare = {
        'Región 1': f'{base_path}new_stellar_mesh_sigma_omega_rho_rho0_lambda_A_propios_1.npz',
        'Región 3': f'{base_path}new_stellar_mesh_sigma_omega_rho_rho0_lambda_A_propios_3.npz',
        'Región 4': f'{base_path}new_stellar_mesh_sigma_omega_rho_rho0_lambda_A_propios_4.npz',
        'Región 6': f'{base_path}new_stellar_mesh_sigma_omega_rho_rho0_lambda_A_propios_6.npz',
    }
    
    # Generar gráfica de comparación
    start_time = time()
    fig = plot_comparison(files_to_compare, dens_min=2.5e-2, dens_max=4.0, rf=30.0, dr=3e-5)
    end_time = time()
    print(f"\nTiempo de ejecución: {end_time - start_time:.2f} segundos")
    plt.show()
