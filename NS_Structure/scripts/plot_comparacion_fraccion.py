"""
Script para comparar energía de enlace y fracción de neutrones
entre diferentes regiones del espacio de parámetros.
"""

import numpy as np
import matplotlib.pyplot as plt
import IsospinEoS as isoEoS
import NSMatterEoS as nsEoS

# Usamos el directorio del archivo
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Constantes físicas en MKS
hbar_MKS = 1.0545718e-34  # J s
c_MKS = 299792458  # m/s
e_MKS = 1.602186021766e-19  # C
proton_mass = 1.6726219e-27  # kg
neutron_mass = 1.6749275e-27  # kg
m_nuc_MKS = (proton_mass + neutron_mass) / 2.0  # kg

# Conversiones útiles
Kg_to_fm11 = c_MKS / hbar_MKS * 1e-15  # kg to fm^-1
MeV_to_fm11 = e_MKS / (hbar_MKS * c_MKS * 1e9)  # MeV to fm^-1

# Constantes en unidades naturales
m_nuc = m_nuc_MKS * Kg_to_fm11  # fm^-1
rho_0_lambda = m_nuc**4 / 2.0


def find_max_mass_params(filepath):
    """
    Encuentra los parámetros que producen la masa máxima en un archivo de malla.
    """
    data = np.load(filepath)
    
    A_sigma_range = data['A_sigma_range']
    A_omega_range = data['A_omega_range']
    params_base = data['params']
    mass_mesh = data['mass_mesh']
    
    # Encontrar el índice de la masa máxima (ignorando NaNs)
    max_idx = np.nanargmax(mass_mesh)
    i, j = np.unravel_index(max_idx, mass_mesh.shape)
    
    A_sigma_max = A_sigma_range[j]
    A_omega_max = A_omega_range[i]
    
    # Construir lista completa de parámetros
    params = [A_sigma_max, A_omega_max, params_base[2], params_base[3], params_base[4]]
    
    return params


def compute_properties(params, n_bind=None, n_frac=None):
    """
    Calcula energía de enlace y fracción de neutrones.
    """
    # Rango para energía de enlace (materia simétrica)
    if n_bind is None:
        n_bind = np.linspace(0.01, 0.5, 200)
    binding_energy = np.zeros_like(n_bind)
    
    # Parámetros para IsospinEoS (agregando t=0 para materia simétrica)
    params_iso = params + [0.0]
    
    for i, n in enumerate(n_bind):
        ener_dimless, _ = isoEoS.energia_presion(n, params_iso, verbose=False)
        ener_dens = ener_dimless * rho_0_lambda
        binding_energy[i] = (ener_dens / n - m_nuc) / MeV_to_fm11

    # Rango para fracción de neutrones (materia de estrella de neutrones)
    if n_frac is None:
        n_frac = np.logspace(-4, 0.5, 200)
    n_p, n_n, n_e = nsEoS.distribucion_especies(n_frac, params)
    neutron_fraction = n_n / n_frac
    
    return n_bind, binding_energy, n_frac, neutron_fraction


def plot_comparison(files_dict, n_bind=None, n_frac=None):
    """
    Compara energía de enlace y fracción de neutrones.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Colores y estilos de línea
    colors = plt.cm.copper_r(np.linspace(0.2, 0.9, len(files_dict)))
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    # Procesar cada archivo
    for idx, (label, filepath) in enumerate(files_dict.items()):
        print(f"Procesando {label}...")
        
        # Encontrar parámetros
        params = find_max_mass_params(filepath)
        
        # Calcular propiedades
        n_bind, binding_energy, n_frac, neutron_fraction = compute_properties(params, n_bind=n_bind, n_frac=n_frac)
        print(max(neutron_fraction))
        # Crear etiqueta con parámetros
        param_label = (f"$A_\\sigma$={params[0]:.1f}, $A_\\omega$={params[1]:.1f}, "
                      f"$A_\\rho$={params[2]:.1f}, \n$b$={params[3]:.2e}, $c$={params[4]:.2e}")
        
        # Estilo
        color = colors[idx]
        ls = line_styles[idx % len(line_styles)]
        
        # Graficar Energía de Enlace
        axes[0].plot(n_bind, binding_energy, color=color, linestyle=ls, linewidth=1.8, label=param_label)
        
        # Graficar Fracción de Neutrones
        axes[1].semilogx(n_frac, neutron_fraction, color=color, linestyle=ls, linewidth=1.8, label=param_label)
        
    # Configurar subplot Energía de Enlace
    axes[0].set_xlabel(r'Densidad bariónica (fm$^{-3}$)', fontsize=14)
    axes[0].set_ylabel(r'B/A (MeV)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(n_bind[0], n_bind[-1])
    # axes[0].set_ylim(-20, 40)
    
    # Agregar eje secundario superior con densidad de masa
    secax0 = axes[0].secondary_xaxis('top', 
                                     functions=(lambda x: x * 1e45 * m_nuc_MKS * 1e-3,
                                               lambda x: x / (1e45 * m_nuc_MKS * 1e-3)))
    secax0.set_xlabel(r'Densidad de masa (g/cm$^3$)', fontsize=14)
    
    # Agregar línea vertical en densidad de saturación
    n_sat = 0.157  # fm^-3
    for i, ax in enumerate([axes[0]]):
        ax.axvline(n_sat, color='black', linestyle='--', linewidth=1, zorder=1)
        x_pos = [n_sat + 0.03 if i == 0 else n_sat + 0.1] 
        ax.text(x_pos[0],
                ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15, r'$n_0$', 
                fontsize=14, ha='center', va='top')
    
    # Configurar subplot Fracción de Neutrones
    axes[1].set_xlabel(r'Densidad bariónica (fm$^{-3}$)', fontsize=14)
    axes[1].set_ylabel(r'$n_n/n_B$', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(n_frac[0], n_frac[-1]*1.1)
    axes[1].set_ylim(0.6, 1.01)
    
    # Agregar eje secundario superior con densidad de masa
    secax1 = axes[1].secondary_xaxis('top',
                                     functions=(lambda x: x * 1e45 * m_nuc_MKS * 1e-3,
                                               lambda x: x / (1e45 * m_nuc_MKS * 1e-3)))
    secax1.set_xlabel(r'Densidad de masa (g/cm$^3$)', fontsize=14)
    
    # Leyendas
    axes[0].legend(fontsize=10, loc='upper left', labelspacing=1.2, framealpha=1)

    # axes[1].legend(fontsize=12, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Definir archivos a comparar
    base_path = '../results/EspacioParametros/'
    files_to_compare = {
        'Región 1': f'{base_path}new_stellar_mesh_sigma_omega_rho_rho0_lambda_A_propios_1.npz',
        'Región 3': f'{base_path}new_stellar_mesh_sigma_omega_rho_rho0_lambda_A_propios_3.npz',
        'Región 4': f'{base_path}new_stellar_mesh_sigma_omega_rho_rho0_lambda_A_propios_4.npz',
        'Región 6': f'{base_path}new_stellar_mesh_sigma_omega_rho_rho0_lambda_A_propios_6.npz',
    }
    
    # Generar gráfica de comparación
    n_bind = np.linspace(0.001, 0.7, 200)
    n_frac = np.logspace(np.log10(0.157), 1, 200)
    fig = plot_comparison(files_to_compare, n_bind=n_bind, n_frac=n_frac)
    plt.show()
