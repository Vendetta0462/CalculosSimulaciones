"""
Script para graficar la descomposición de la energía de enlace por nucleón
en sus componentes: sigma (escalar), omega (vectorial) y cinética.

Para materia nuclear simétrica (t=0), el campo rho no contribuye.
"""

import numpy as np
import matplotlib.pyplot as plt
import IsospinEoS as isoEoS

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
pi = np.pi


def compute_energy_components(n_barion, params):
    """
    Calcula las componentes de la energía de enlace por nucleón.
    
    Parameters:
    -----------
    n_barion : float
        Densidad bariónica en fm^-3
    params : list
        [A_sigma, A_omega, A_rho, b, c, t]
        
    Returns:
    --------
    E_sigma : float
        Contribución del campo sigma (MeV)
    E_omega : float
        Contribución del campo omega (MeV)
    E_kin : float
        Contribución cinética (MeV)
    E_total : float
        Energía total de enlace por nucleón (MeV)
    """
    A_sigma, A_omega, A_rho, b, c, t = params
    
    # Resolver ecuación de autoconsistencia
    x_sigma = isoEoS.sol_x_sigma(n_barion, params, verbose=False)
    x_f = (1.0 / m_nuc) * (3.0 * pi**2 * n_barion / 2.0)**(1/3)
    
    # Para materia simétrica (t=0), x_nF = x_pF = x_f
    x_nF = x_f * (1.0 - t)**(1/3)
    x_pF = x_f * (1.0 + t)**(1/3)
    
    # Calcular integrales para energía cinética
    if x_nF > 0:
        raiz_n = np.sqrt(x_nF**2 + x_sigma**2)
        termino_arctanh_n = x_sigma**4 * np.arctanh(x_nF / raiz_n)
        integral_energia_n = (x_nF * raiz_n * (2.0 * x_nF**2 + x_sigma**2) - termino_arctanh_n) / 8.0
    else:
        integral_energia_n = 0
    
    if x_pF > 0:
        raiz_p = np.sqrt(x_pF**2 + x_sigma**2)
        termino_arctanh_p = x_sigma**4 * np.arctanh(x_pF / raiz_p)
        integral_energia_p = (x_pF * raiz_p * (2.0 * x_pF**2 + x_sigma**2) - termino_arctanh_p) / 8.0
    else:
        integral_energia_p = 0
    
    # Contribución del campo sigma (escalar)
    termino_sigma = (1.0 - x_sigma)**2 * (1.0/A_sigma + 2.0/3.0*b*(1.0-x_sigma) + 0.5*c*(1-x_sigma)**2)
    E_sigma_adim = termino_sigma
    
    # Contribución del campo omega (vectorial)
    termino_omega = (A_omega + 0.25*A_rho*t**2) * (4.0*x_f**6/(9.0*pi**4))
    E_omega_adim = termino_omega
    
    # Contribución cinética
    E_kin_adim = 2.0/(pi**2) * (integral_energia_n + integral_energia_p)
    
    # Energía total adimensional
    energia_total_adim = E_sigma_adim + E_omega_adim + E_kin_adim
    
    # Convertir a densidad de energía dimensional
    energia_total_dens = energia_total_adim * rho_0_lambda
    E_sigma_dens = E_sigma_adim * rho_0_lambda
    E_omega_dens = E_omega_adim * rho_0_lambda
    E_kin_dens = E_kin_adim * rho_0_lambda
    
    # Energía de enlace por nucleón (restar masa del nucleón)
    E_total = (energia_total_dens / n_barion - m_nuc) / MeV_to_fm11
    E_sigma = (E_sigma_dens / n_barion) / MeV_to_fm11
    E_omega = (E_omega_dens / n_barion) / MeV_to_fm11
    E_kin = (E_kin_dens / n_barion - m_nuc) / MeV_to_fm11
    # E_kin = (E_kin_dens / n_barion) / MeV_to_fm11
    
    return E_sigma+E_kin, E_omega, E_kin, E_total


def plot_energy_decomposition(params, n_range=None):
    """
    Grafica la descomposición de la energía de enlace.
    
    Parameters:
    -----------
    params : list
        [A_sigma, A_omega, A_rho, b, c, t] parámetros del modelo
    n_range : array, optional
        Rango de densidades bariónicas (fm^-3)
    """
    if n_range is None:
        n_range = np.linspace(0.001, 0.7, 300)
    
    # Arrays para almacenar resultados
    E_sigma_arr = np.zeros_like(n_range)
    E_omega_arr = np.zeros_like(n_range)
    E_kin_arr = np.zeros_like(n_range)
    E_total_arr = np.zeros_like(n_range)
    
    # Calcular componentes para cada densidad
    for i, n in enumerate(n_range):
        E_sigma, E_omega, E_kin, E_total = compute_energy_components(n, params)
        E_sigma_arr[i] = E_sigma
        E_omega_arr[i] = E_omega
        E_kin_arr[i] = E_kin
        E_total_arr[i] = E_total
    
    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    # Colormap y estilos
    # colors = plt.cm.copper_r(np.linspace(0.2, 0.9, 4))
    colors = ["#914098", "#852B17", "#3E7C17", "k"]
    
    # Graficar componentes
    ax.plot(n_range, E_sigma_arr, color=colors[0], linestyle='-.', linewidth=1.8, 
            label=r'$\sigma$')
    ax.plot(n_range, E_omega_arr, color=colors[1], linestyle='--', linewidth=1.8,
            label=r'$\omega$')
    # ax.plot(n_range, E_kin_arr, color=colors[2], linestyle=':', linewidth=1.8,
    #         label=r'Cinética')
    ax.plot(n_range, E_total_arr, color=colors[3], linestyle='-', linewidth=2,
            label='Total', zorder=5)
    
    # Configurar ejes
    ax.set_xlabel(r'Densidad bariónica (fm$^{-3}$)', fontsize=14)
    ax.set_ylabel(r'B/A (MeV)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(n_range[0], n_range[-1])
    # ax.set_ylim(min(E_total_arr)*1.2, 400)
    
    # Eje secundario con densidad de masa
    secax = ax.secondary_xaxis('top',
                               functions=(lambda x: x * 1e45 * m_nuc_MKS * 1e-3,
                                         lambda x: x / (1e45 * m_nuc_MKS * 1e-3)))
    secax.set_xlabel(r'Densidad de masa (g/cm$^3$)', fontsize=14)
    
    # Línea vertical en densidad de saturación
    n_sat = 0.157  # fm^-3
    ax.axvline(n_sat, color='black', linestyle='--', linewidth=1, zorder=1)
    ax.text(n_sat + 0.02, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15,
            r'$n_0$', fontsize=14, ha='center', va='top')
    
    # Línea horizontal en E=0
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    # Leyenda
    ax.legend(fontsize=12, loc='best', framealpha=1)
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Parámetros del modelo (Glendenning)
    A_sigma = 12.684 * m_nuc**2
    A_omega = 7.148 * m_nuc**2
    A_rho = 4.410 * m_nuc**2
    b = 5.610e-3
    c = -6.986e-3
    t = 0.0  # Materia simétrica
    
    params = [A_sigma, A_omega, A_rho, b, c, t]
    
    # Rango de densidades
    n_range = np.linspace(0.001, 0.7, 300)
    
    # Generar gráfica
    fig = plot_energy_decomposition(params, n_range)
    plt.show()
