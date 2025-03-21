import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

# Definimos las constantes físicas
c = 299792458 # Velocidad de la luz en m/s
G = 6.67430e-11 # Constante de gravitación universal en m^3/kg/s^2
pi = np.pi # Constante pi
m_nuc_MKS = 1.66053906660e-27 # Masa de un nucleón en kg

def adimensional_to_fisico(sol_fin, P_central, r_fin, rho0):
    """
    Converts dimensionless quantities to physical quantities IN MKS for the TOV equations.

    Parameters
    ----------
    sol_fin : list
        A list containing the final values of the dimensionless solutions (P, m, phi, rho).
    P_central : float
        The central pressure in dimensionless units.
    r_fin : float
        The final radius in dimensionless units.
    rho0 : float
        The central density in physical units.

    Returns
    -------
    list
        A list containing the physical quantities:
        - r_fisico (float): The physical radius.
        - m_fisico (float): The physical mass.
        - rho_fisico (float): The physical density.
        - P_fisico (float): The physical pressure.
        - phi_fisico (float): The physical metric potential.
    """
    P, m, phi, rho = sol_fin
    
    # Obtenemos las constantes de adimensionalización
    P0 = rho0
    R = np.sqrt(c**4/(4*pi*G*rho0))
    M = 4*pi*R**3*rho0/c**2

    # Convertimos las cantidades adimensionales a cantidades físicas
    P_fisico = P0*P # Presión final en Pa
    m_fisico = M*m # Masa en kg
    phi_fisico = phi # Potencial adimensional
    rho_fisico = rho*rho0 # Densidad final de energía en J/m^3
    r_fisico = R*r_fin # Radio en m

    return [r_fisico, m_fisico, rho_fisico, P_fisico, phi_fisico]

def newtonianas(sol, r, rho_P):
    """
    Calculate the derivatives of pressure, mass, and dimensionless potential using Newtonian equations.

    Parameters
    ----------
    sol : list or tuple
        A list or tuple containing the dimensionless pressure (P), mass (m), and dimensionless potential (phi).
    r : float
        The radial coordinate.
    rho_P : function
        A function that takes pressure (P) as input and returns the dimensionless energy density (rho).

    Returns
    -------
    list
        A list containing the derivatives of pressure (dP_dr), mass (dm_dr), and dimensionless potential (dphi_dr) with respect to the radial coordinate.
    """
    P, m, phi = sol # Presion, masa y potencial adimensionales

    # Usamos la ecuación de estado para la densidad de energía
    rho = rho_P(P) # Densidad de energia adimensional

    # Definimos las derivadas
    dm_dr = r**2 * rho
    dP_dr = -m*rho/r**2
    dphi_dr = m/r**2

    # Devolvemos las derivadas
    return [dP_dr, dm_dr, dphi_dr]

def relativistas(sol, r, rho_P):
    """
    Calculate the derivatives of pressure, mass, and dimensionless potential for a relativistic star using the Tolman-Oppenheimer-Volkoff (TOV) equations.

    Parameters
    ----------
    sol : list
        A list containing the current values of pressure (P), mass (m), and dimensionless potential (phi).
    r : float
        The radial coordinate.
    rho_P : function
        A function that takes pressure (P) as input and returns the corresponding energy density (rho).

    Returns
    -------
    list
        A list containing the derivatives of pressure (dP_dr), mass (dm_dr), and dimensionless potential (dphi_dr) with respect to the radial coordinate.
    """
    P, m, phi = sol # Presion, masa y potencial adimensionales

    # Usamos la ecuación de estado para la densidad de energía
    rho = rho_P(P) # Densidad de energia adimensional

    # Definimos las derivadas
    dm_dr = r**2 * rho
    dP_dr = -m*rho/r**2*(1+P/rho)*(1+r**3*P/m)/(1-2*m/r)
    dphi_dr = -dP_dr/(rho + P)

    # Devolvemos las derivadas
    return [dP_dr, dm_dr, dphi_dr]

def integrador(rf, dr, rho0, rho_P, P_central, sistema='GR', sol_completa=False, densidad_limite=None):
    """
    Integrates the TOV equations to solve for the structure of a neutron star.

    Parameters
    ----------
    rf : float
        Final radius for the integration (dimensionless).
    dr : float
        Step size for the integration (dimensionless).
    rho0 : float
        Central energy density (in MKS units).
    rho_P : function
        Equation of state function that relates density to pressure (dimensionless).
    P_central : float
        Central pressure (dimensionless).
    sistema : str, optional
        The system of equations to use ('GR' for General Relativity or 'Newt' for Newtonian). Default is 'GR'.
    sol_completa : bool, optional
        If True, returns the complete solution. If False, returns only the final quantities. Default is False.
    densidad_limite : float, optional
        The energy density limit for the star (dimensionless). If None, the limit is set when the pressure reaches zero. Default is None.

    Returns
    -------
    tuple
        If sol_completa is False, returns a tuple with the final physical quantities (r_fisico, m_fisico, rho_fisico, P_fisico, phi_fisico) in MKS units.
        If sol_completa is True, returns a tuple with the final physical quantities in MKS units, the complete solution array, and the radius array (both dimensionless).
    """
    # Malla de integración para ambos sistemas
    N = int(rf/dr) # Número de puntos
    r = np.linspace(dr, rf, N) # Puntos de integración

    # Definimos las condiciones iniciales en r = dr (para evitar la singularidad en r = 0)
    # estas son cantidades adimensionales, así que rho(0) = 1 y m(0) = 0
    rhoc = 1.0
    m0 = rhoc*dr**3/3 # m = 0 + 0r + 0r^2/2 + 2rho(0)r^3/6 ...
    if sistema == 'Newt':
        ecuaciones = newtonianas
        P0 = P_central- rhoc * dr**2 / 6.0 # P = P(0) + 0r -1/3*r^2/2 ...
        phi0 = 1 + rhoc**2*dr**2/6 # phi = 1 + 0r + 1/3*r^2/2 ...
    elif sistema == 'GR':
        ecuaciones = relativistas
        P0 = P_central- ( 3.0*P_central**2 + 4.0*P_central*rhoc + rhoc**2)*dr**2/6.0 # P = P(0) + 0r -1/3*(3P(0)^2+4P(0)+1)*r^2/2 ...
        phi0 = 1 + ( rhoc/3.0 + P_central ) * dr**2 / 2.0 # phi = 1 + 0r + (rho(0)/3 + P(0))*r^2/2 ...

    # Resolvemos el sistema de ecuaciones
    sol = odeint(ecuaciones, [P0, m0, phi0], r, args=(rho_P,)).T

    # Obtenemos la densidad a partir de las presiones
    P, m, phi = sol
    rho = rho_P(P)

    # Buscamos el radio límite de la estrella
    lim = len(r)-1
    if densidad_limite == None:
        # la condicion debe ser con P, no siempre cuando P=0, rho=0
        for i in range(len(P)):
            if P[i] <= 0:
                lim = i-1
                break
    else:
        # la condicion para el limite de nuestra estrella es la densidad de energia adimensional definida por densidad_limite
        for i in range(len(r)):
            if rho[i] <= densidad_limite:
                lim = i-1
                break

    # Obtenemos las cantidades finales
    sol_fin = [P[lim], m[lim], phi[lim], rho[lim]]

    if sol_completa:
        return adimensional_to_fisico(sol_fin, P_central, r[lim], rho0), np.append(sol[:, :lim], [rho[:lim]], axis=0), r[:lim]
    else:
        # Convertimos las cantidades adimensionales a cantidades físicas
        return adimensional_to_fisico(sol_fin, P_central, r[lim], rho0)
    
def grafica_masa_radio(radios, masas, densidades_masa):
    """
    Plots the mass-radius and mass-central density relations for neutron stars.

    Parameters
    ----------
    radios : list
        List of radii of neutron stars.
    masas : list
        List of masses of neutron stars.
    densidades_masa : list
        List of central densities of neutron stars.

    Returns
    -------
    None
    """

    # Graficamos las relaciones masa radio y masa densidad en masas solares
    masasolar = 1.989e30  # Masa solar en kg
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Creamos un colormap personalizado para los puntos
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["cornflowerblue", "darkorchid"])
    norm = mcolors.Normalize(vmin=0, vmax=len(radios) - 1)
    colors = [cmap(norm(i)) for i in range(len(radios))]

    # Graficamos la relación masa-radio
    for i in range(len(radios)):
        ax[0].plot(radios[i] / 1e3, masas[i] / masasolar, 'o', color=colors[i], linewidth=2)
    ax[0].set_xlabel(r'$R$ (km)', fontsize=16)
    ax[0].set_ylabel(r'$M$ ($M_\odot$)', fontsize=16)
    ax[0].set_title(r'Relación Masa-Radio', fontsize=18)
    ax[0].tick_params(axis='both', which='both', direction='in', right=True, top=True)
    # Punto en la masa maxima
    ax[0].plot(radios[np.argmax(masas)] / 1e3, masas.max() / masasolar, 'o', color='red', markeredgewidth=1, markeredgecolor='black')
    ax[0].grid()

    # Graficamos la relación masa-densidad central
    for i in range(len(densidades_masa)):
        ax[1].semilogx(densidades_masa[i], masas[i] / masasolar, 'o', color=colors[i], linewidth=2)
    ax[1].set_xlabel(r'$\rho_0^m$ (g/cm$^3$)', fontsize=16)
    ax[1].set_title(r'Relación Masa-Densidad Central', fontsize=18)
    ax[1].tick_params(axis='both', which='both', direction='in', right=True, top=True)
    # Punto en la masa maxima
    ax[1].plot(densidades_masa[np.argmax(masas)], masas.max() / masasolar, 'o', color='red', markeredgewidth=1, markeredgecolor='black')
    ax[1].grid()

    print("Masa máxima:", format(masas.max() / masasolar, "2.3f"), "M_sun para rho0_m =", format(densidades_masa[np.argmax(masas)], "2.3e"), "g/cm^3")

    plt.tight_layout()
    plt.show()

def graficar_solucion(rf, dr, rho0, rho_P, P_central, sistema='GR', densidad_limite=None):
    """
    Plots the solution of the TOV equations.

    Parameters
    ----------
    rf : float
        Final radius for the integration (dimensionless).
    dr : float
        Step size for the integration (dimensionless).
    rho0 : float
        Central energy density (in MKS units).
    rho_P : function
        Equation of state function that relates density to pressure (dimensionless).
    P_central : float
        Central pressure (dimensionless).
    sistema : str, optional
        The system of equations to use ('GR' for General Relativity or 'Newt' for Newtonian). Default is 'GR'.
    densidad_limite : float, optional
        The energy density limit for the star (dimensionless). If None, the limit is set when the pressure reaches zero. Default is None.

    Returns
    -------
    None
    """
    # Resolvemos el sistema de ecuaciones
    sol, sol_completa, r = integrador(rf, dr, rho0, rho_P, P_central, sistema, sol_completa=True, densidad_limite=densidad_limite)
    
    # Graficamos la solución
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    
    # Graficamos la presión
    ax[0, 0].plot(r, sol_completa[0], 'o-', color='cornflowerblue', linewidth=2)
    ax[0, 0].set_xlabel(r'$r$', fontsize=14)
    ax[0, 0].set_ylabel(r'$P$', fontsize=14)
    ax[0, 0].set_title(r'Presión', fontsize=15)
    ax[0, 0].tick_params(axis='both', which='both', direction='in', right=True, top=True)
    
    # Graficamos la masa
    ax[0, 1].plot(r, sol_completa[1], 'o-', color='cornflowerblue', linewidth=2)
    ax[0, 1].set_xlabel(r'$r$', fontsize=14)
    ax[0, 1].set_ylabel(r'$m$', fontsize=14)
    ax[0, 1].set_title(r'Masa', fontsize=15)
    ax[0, 1].tick_params(axis='both', which='both', direction='in', right=True, top=True)
    
    # Graficamos el potencial métrico
    ax[1, 0].plot(r, sol_completa[2], 'o-', color='cornflowerblue', linewidth=2)
    ax[1, 0].set_xlabel(r'$r$', fontsize=14)
    ax[1, 0].set_ylabel(r'$\phi$', fontsize=14)
    ax[1, 0].set_title(r'Potencial Métrico', fontsize=15)
    ax[1, 0].tick_params(axis='both', which='both', direction='in', right=True, top=True)
    
    # Graficamos la densidad
    ax[1, 1].plot(r, sol_completa[3], 'o-', color='cornflowerblue', linewidth=2)
    ax[1, 1].set_xlabel(r'$r$', fontsize=14)
    ax[1, 1].set_ylabel(r'$\rho$', fontsize=14)
    ax[1, 1].set_title(r'Densidad', fontsize=15)
    ax[1, 1].tick_params(axis='both', which='both', direction='in', right=True, top=True)
    
    fig.suptitle(r'Solución de las ecuaciones TOV', fontsize=18)
    
    plt.tight_layout()
    plt.show()
    
def gas_neutrones(rho0_m, rho0_m_max=1e17, extrapolate=False):
    """
    Defines an equation of state for neutrons, protons, and electrons as an example.

    Parameters
    ----------
    rho0_m : float
        Central mass density in g/cm^3.
    rho0_m_max : float, optional
        Maximum mass density in g/cm^3. Default is 1e17.
    extrapolate : bool, optional
        If True, allows extrapolation beyond the provided data. Default is False.

    Returns
    -------
    tuple
        A tuple containing the interpolation function for pressure and density, the central energy density, and the central pressure.
    """
    m_n = 1.674927471e-27 # Masa del neutrón en kg
    h = 6.62607015e-34 # Constante de Planck en J s
    
    # Definimos las constantes A y B
    A = pi*m_n**4*c**5/(3*h**3) 
    B = 8*pi*m_n**4*c**3/(3*h**3)
    
    x_0 = ((rho0_m * 1e3)/B)**(1/3)

    # Hallamos la densidad de energía central
    fs_0 = x_0*np.sqrt(1+x_0**2)*(2*x_0**2-3)+3*np.log(x_0+np.sqrt(1+x_0**2))
    rho0 = (c**2*B*x_0**3 + A*(8*x_0**3*(np.sqrt(1+x_0**2)-1)-fs_0))
    
    # Definimos el espacio de parametros x
    x = np.linspace(-4, np.log10(((rho0_m_max * 1e3)/B)**(1/3)), 1000)
    x = np.power(10, x)

    # Definimos la función f(x)
    fs = x*np.sqrt(1+x**2)*(2*x**2-3)+3*np.log(x+np.sqrt(1+x**2))
    
    # Hallamos la presión adimensional
    Ps = A*fs/rho0
        
    # Hallamos la densidad de energía adimensional
    densidades = ( c**2*B*x**3 + A*(8*x**3*(np.sqrt(1+x**2)-1)-fs) )/rho0
    
    # Hallamos la presión central
    P_central = interp1d(densidades, Ps)(1)

    # Devolvemos la función de interpolación
    if extrapolate:
        return interp1d(Ps, densidades, fill_value='extrapolate'), rho0, P_central
    else:
        return interp1d(Ps, densidades), rho0, P_central