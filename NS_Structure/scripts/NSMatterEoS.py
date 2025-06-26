# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

#-----------------------------------------------------------------------
# DEFINICIONES
#-----------------------------------------------------------------------

# Definimos las constantes necesarias en MKS
hbar_MKS = 1.0545718e-34 # J s
c_MKS = 299792458 # m/s
G_MKS = 6.67430e-11 # m^3/kg/s^2
pi = np.pi
proton_mass = 1.6726219e-27 # kg
neutron_mass = 1.6749275e-27 # kg
electron_mass = 9.1093837e-31 # kg
m_nuc_MKS = (proton_mass + neutron_mass)/2.0 # kg
e_MKS = 1.6021766e-19 # J

# Algunas converciones útiles (multiplicar al primero para obtener el segundo)
Kg_to_fm11 = c_MKS/hbar_MKS*1e-15 # kg to fm^-1
MeV_to_fm11 = e_MKS/(hbar_MKS*c_MKS*1e9) # MeV to fm^-1
MeVfm_to_Jm  = 1e51*e_MKS # MeV/fm to J/m

# Definimos las constantes necesarias en unidades naturales
m_nuc = m_nuc_MKS * Kg_to_fm11 # fm^-1
m_e = electron_mass * Kg_to_fm11 # fm^-1

# Damos valores a las constantes (Glendenning) (constantes tilde cuadradas)
A_sigma = 12.684*m_nuc**2 # Walecka: 266.9, 357.4
A_omega =  7.148*m_nuc**2 # Walecka: 195.7, 273.8
A_rho   =  4.410*m_nuc**2 # Nuevo parámetro para el campo rho
b       =  5.610e-3
c       = -6.986e-3  


#-------------------------------------------------------------------------
# ECUACIÓN DE ESTADO
#-------------------------------------------------------------------------

def ecuaciones_autoconsistencia(variables, n_barion, params=[A_sigma, A_omega, A_rho, b, c]):
    """
    Calcula la ecuación de autoconsistencia (ecuación de campo sigma) para el modelosigma-omega-rho
    con autointeracciones del campo escalar y la condición de equilibrio beta en la estrella.

    Args:
        variables (array): Valores para el campo escalar sigma y el momento de fermi del neutron [x_sigma, x_nF].
        n_barion (float): Densidad bariónica en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.

    Returns:
        array: Valores de la función de autoconsistencia y equilibrio beta que deben ser cero para encontrar la solución.
    """
    A_sigma, _, _, b, c = params
    x_sigma, x_nF = variables # Masa efectiva relativa del nucleón y momento de Fermi del neutron
    
    # Momento de Fermi del proton (y electron, x_pF=x_eF)
    # m_charge = (proton_mass+electron_mass)*Kg_to_fm11
    # x_pF = 3.0*pi**2/2.0 * (n_barion/(m_nuc**3*m_charge) + 2.0/(3.0*pi**2)*(neutron_mass*Kg_to_fm11)/m_charge*x_nF**3)**(1/3) # Momento de Fermi del protón (y electrón)
    x_pF = (3.0*pi**2/2.0*n_barion/m_nuc**3 - x_nF**3)**(1/3) # Momento de Fermi del protón (y electrón)
    
    # Integrales separadas para neutrones y protones
    if x_nF > 0:
        raiz_n = np.sqrt(x_nF**2 + x_sigma**2)
        integral_n = 0.5 * x_sigma* (x_nF * raiz_n - x_sigma**2 * np.arctanh(x_nF / raiz_n))
    else:
        integral_n = 0
        
    if x_pF > 0:
        raiz_p = np.sqrt(x_pF**2 + x_sigma**2)
        integral_p = 0.5 * x_sigma * (x_pF * raiz_p - x_sigma**2 * np.arctanh(x_pF / raiz_p))
    else:
        integral_p = 0
    
    integral_total = integral_n + integral_p
    
        # Ecuación de equilibrio beta (=0)
    if x_pF > 0 and x_nF > 0:
        equilibrio_beta = raiz_n - raiz_p - np.sqrt(x_pF**2 + (electron_mass*Kg_to_fm11)**2/m_nuc**2) - 1/(3*pi**2)*A_rho*(x_pF**3 - x_nF**3)
    else:
        equilibrio_beta = 100 
    # Ecuación de campo del meson escalar (=0)
    campo_sigma = (1.0 - x_sigma) - A_sigma * ( integral_total/(pi**2) - b*(1-x_sigma)**2 - c*(1-x_sigma)**3 )
    
    # Retornamos las ecuaciones de autoconsistencia y equilibrio beta
    return np.array([campo_sigma, equilibrio_beta])

def sol_x_sigma_x_nF(n_barion, params=[A_sigma, A_omega, A_rho, b, c], x0=[0.5, 0.2], print_solution=False):
    """
    Resuelve las ecuaciones de autoconsistencia para un valor dado de densidad bariónica, encontrando raíces.

    Args:
        n_barion (float): Densidad bariónica en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.
        x0 (list): Valor inicial para la solución [x_sigma, x_nF].
        print_solution (bool): Si es True, imprime detalles de la solución encontrada.

    Returns:
        float: Solución para el campo escalar sigma.
    """
    
    # Graficamos la superficie de ecuaciones_autoconsistencia para n_barion
    if print_solution:
        x_sigma_plot = np.linspace(1e-3, 1, 100)
        x_nF_plot = np.linspace(1e-3, 1, 100)
        X_sigma, X_nF = np.meshgrid(x_sigma_plot, x_nF_plot)
        Z1 = np.zeros_like(X_sigma)
        Z2 = np.zeros_like(X_sigma)
        for i in range(len(x_sigma_plot)):
            for j in range(len(x_nF_plot)):
                sol_ = ecuaciones_autoconsistencia([X_sigma[i, j], X_nF[i, j]], n_barion, params)
                Z1[i, j] = sol_[0]  # Ecuación de campo del meson escalar
                Z2[i, j] = sol_[1]  # Ecuación de equilibrio beta

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Primer subplot: ecuación de campo del mesón escalar
        c1 = axs[0].contourf(X_sigma, X_nF, Z1, levels=np.linspace(-1, 1, 21), cmap='coolwarm', alpha=0.7)
        fig.colorbar(c1, ax=axs[0], label='Ecuación de autoconsistencia')
        axs[0].set_xlabel('x_sigma')
        axs[0].set_ylabel('x_nF')
        axs[0].set_title(f'Ecuación de campo escalar\nn_barion = {n_barion}')
        axs[0].axhline(y=x0[1], color='k', linestyle='--', label='x_nF inicial')
        axs[0].axvline(x=x0[0], color='k', linestyle='--', label='x_sigma inicial')
        axs[0].legend()

        # Segundo subplot: ecuación de equilibrio beta
        c2 = axs[1].contourf(X_sigma, X_nF, Z2, levels=np.linspace(np.min(Z2), np.max(Z2), 21), cmap='coolwarm', alpha=0.7)
        fig.colorbar(c2, ax=axs[1], label='Ecuación de equilibrio beta')
        axs[1].set_xlabel('x_sigma')
        axs[1].set_ylabel('x_nF')
        axs[1].set_title(f'Ecuación de equilibrio beta\nn_barion = {n_barion}')
        axs[1].axhline(y=x0[1], color='k', linestyle='--', label='x_nF inicial')
        axs[1].axvline(x=x0[0], color='k', linestyle='--', label='x_sigma inicial')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    solution = fsolve(ecuaciones_autoconsistencia, x0, args=(n_barion, params), full_output=True)
    if solution[2] != 1:
        print("No se encontró solución para n_barion = ", n_barion)
        if print_solution:
            print("Error:", solution[3])
            print("Detalles de la solución:", solution[1])
        return [1, 1]
    else:
        if print_solution:
            print("Solución encontrada para n_barion =", n_barion, ":", solution[0])
        return solution[0]

def energia_presion(n_barion, params=[A_sigma, A_omega, A_rho, b, c]):
    """
    Calcula la energía y la presión para una densidad bariónica dada.

    Args:
        n_barion (float): Densidad bariónica en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.

    Returns:
        tuple: Energía y presión calculadas en unidades de lambda/2 = m_nuc^4/2 (unidades naturales).
    """
    A_sigma, A_omega, A_rho, b, c = params
    x_sigma, x_nF = sol_x_sigma_x_nF(n_barion, params)
    # x_f = (1.0/m_nuc)*(3.0*pi**2*n_barion/2.0)**(1/3)
    
    # Momento de Fermi para protón
    x_pF = (3.0*pi**2/2.0*n_barion/m_nuc**3 - x_nF**3)**(1/3)  # Momento de Fermi del protón (y electrón)
   
    # Integrales para neutrones
    if x_nF > 0:
        raiz_n = np.sqrt(x_nF**2 + x_sigma**2)
        termino_arctanh_n = x_sigma**4 * np.arctanh(x_nF / raiz_n)
        integral_energia_n = (x_nF * raiz_n * (2.0 * x_nF**2 + x_sigma**2) - termino_arctanh_n) / 8.0
        integral_presion_n = (x_nF * raiz_n * (2.0 * x_nF**2 - 3.0 * x_sigma**2) + 3.0 * termino_arctanh_n) / 8.0
    else:
        integral_energia_n = 0
        integral_presion_n = 0
    
    # Integrales para protones
    m_e_adim = electron_mass*Kg_to_fm11 / m_nuc
    if x_pF > 0:
        raiz_p = np.sqrt(x_pF**2 + x_sigma**2)
        raiz_e = np.sqrt(x_pF**2 + m_e_adim**2)
        termino_arctanh_p = x_sigma**4 * np.arctanh(x_pF / raiz_p)
        termino_arctanh_e = m_e_adim**4 * np.arctanh(x_pF / raiz_e)
        integral_energia_p = (x_pF * raiz_p * (2.0 * x_pF**2 + x_sigma**2) - termino_arctanh_p) / 8.0
        integral_presion_p = (x_pF * raiz_p * (2.0 * x_pF**2 - 3.0 * x_sigma**2) + 3.0 * termino_arctanh_p) / 8.0
        integral_energia_e = (x_pF * raiz_e * (2.0 * x_pF**2 + m_e_adim**2) - termino_arctanh_e) / 8.0
        integral_presion_e = (x_pF * raiz_e * (2.0 * x_pF**2 - 3.0 * m_e_adim**2) + 3.0 * termino_arctanh_e) / 8.0
    else:
        integral_energia_p = 0
        integral_presion_p = 0
        integral_energia_e = 0
        integral_presion_e = 0
    
    # Términos de los campos
    termino_sigma = (1.0-x_sigma)**2 * (1.0/A_sigma + 2.0/3.0*b*(1.0-x_sigma) + 0.5*c*(1-x_sigma)**2)
    termino_omega_rho = A_omega*n_barion**2/m_nuc**6 + 1/(9*pi**4)*A_rho*(x_pF**3 - x_nF**3)**2
   
    
    # Energía y presión totales
    energia = (termino_sigma + termino_omega_rho + 2.0/(pi**2)*(integral_energia_n + integral_energia_p + integral_energia_e))
    presion = (-termino_sigma + termino_omega_rho + 2.0/(3.0*pi**2)*(integral_presion_n + integral_presion_p + integral_presion_e))
    
    return energia, presion

def EoS(n_barions, params=[A_sigma, A_omega, A_rho, b, c]):
    """
    Calcula la ecuación de estado (EoS) para un rango de densidades bariónicas.

    Args:
        n_barions (array-like): Array de densidades bariónicas en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.

    Returns:
        tuple: Interpolación cúbica de la presión y energía, arrays de presiones, energías, densidades y el índice de cambio de signo en la presión.
    """
    energias = np.array([])
    presiones = np.array([])
    n_sirve = np.array([])
    
    for n in n_barions:
        energia, presion = energia_presion(n, params)
        energias = np.append(energias, energia)
        presiones = np.append(presiones, presion)
        n_sirve = np.append(n_sirve, n)
        
    presion_cambio = 0
    for i in range(len(presiones)-1):
        if presiones[i]<0 and presiones[i+1]>0:
            presion_cambio = i+1
            break
        
    return CubicSpline(presiones[presion_cambio:], energias[presion_cambio:]), presiones, energias, n_sirve, presion_cambio

#-----------------------------------------------------------------------
# CALCULO DE PROPIEDADES
#-----------------------------------------------------------------------

# Calculado recientemente
def modulo_compresion(n_sat, params=[A_sigma, A_omega, A_rho, b, c]):
    """
    Calcula el módulo de compresión para una densidad de saturación dada 
    en el modelo sigma-omega-rho con isospin.

    Args:
        n_sat (float): Densidad de saturación en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.

    Returns:
        float: Módulo de compresión en MeV.
    """
    A_sigma, A_omega, A_rho, b, c = params
    x_sigma = sol_x_sigma_x_nF(n_sat, params)
    x_f = (1.0/m_nuc)*(3.0*pi**2*n_sat/2.0)**(1/3)
    
    # Momentos de Fermi para neutrones y protones
    x_nF = x_f * (1.0 )**(1/3)  # neutrones
    x_pF = x_f * (1.0 )**(1/3)  # protones
    
    # Integrales y términos separados para neutrones y protones
    if x_nF > 0:
        raiz_n = np.sqrt(x_nF**2 + x_sigma**2)
        integral_n = 0.5*((x_nF**3 + 3.0*x_nF*x_sigma**2)/raiz_n - 3.0*x_sigma**2*np.arctanh(x_nF/raiz_n))
        factor_interno_n = x_sigma**2*x_nF**3/raiz_n
    else:
        integral_n = 0
        factor_interno_n = 0
        
    if x_pF > 0:
        raiz_p = np.sqrt(x_pF**2 + x_sigma**2)
        integral_p = 0.5*((x_pF**3 + 3.0*x_pF*x_sigma**2)/raiz_p - 3.0*x_sigma**2*np.arctanh(x_pF/raiz_p))
        factor_interno_p = x_sigma**2*x_pF**3/raiz_p
    else:
        integral_p = 0
        factor_interno_p = 0
    
    integral_total = integral_n + integral_p
    factor_interno_total = factor_interno_n + factor_interno_p
    
    # Factor F con las autointeracciones del campo escalar
    F = 1.0 + A_sigma*(1.0/pi**2*integral_total + 2.0*b*(1-x_sigma) + 3.0*c*(1-x_sigma)**2)
    
    # Término vectorial con contribuciones de omega y rho
    termino_vectorial = (2.0*A_omega + 0.5*A_rho*t**2)*x_f**3/pi**2
    
    # Término escalar con derivadas respecto a la densidad
    if x_nF > 0:
        termino_escalar_n = (x_nF/x_f)**3 * (x_nF**2 - A_sigma/(pi**2*F) * factor_interno_total)/raiz_n
    else:
        termino_escalar_n = 0
        
    if x_pF > 0:
        termino_escalar_p = (x_pF/x_f)**3 * (x_pF**2 - A_sigma/(pi**2*F) * factor_interno_total)/raiz_p
    else:
        termino_escalar_p = 0
    
    termino_escalar = termino_escalar_n + termino_escalar_p
    
    K = 3.0*m_nuc*(termino_vectorial + 0.5*termino_escalar)
    
    return K/MeV_to_fm11 # Módulo de compresión en MeV


def coeficiente_simetria(n_sat, params=[A_sigma, A_omega, A_rho, b, c]):
    """
    Calcula el coeficiente de energía de simetría para una densidad de saturación dada.
    
    Args:
        n_sat (float): Densidad de saturación en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.
        
    Returns:
        float: Coeficiente de energía de simetría en MeV.
    """
    _, _, A_rho, _, _, _ = params
    x_sigma = sol_x_sigma_x_nF(n_sat, params)
    x_f = (1.0/m_nuc)*(3.0*pi**2*n_sat/2.0)**(1/3)
    
    # Para materia simétrica (t=0), ambas especies tienen el mismo momento de Fermi
    raiz = np.sqrt(x_f**2 + x_sigma**2)
    a_sym = m_nuc*(x_f**2/(6.0*raiz) + A_rho*x_f**3/(12.0*pi**2))
    
    return a_sym/MeV_to_fm11 # Coeficiente de energía de simetría en MeV

def calculate_properties(n_prove, params):
    """
    Calcula las propiedades de saturación del modelo dado un conjunto de parámetros.
    
    Args:
        n_prove (array-like): Array de densidades bariónicas en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.
        
    Returns:
        props (array): Array con las propiedades de saturación [densidad de saturación, energía de enlace, módulo de compresión, coeficiente de simetría] en fm^-3 y MeV.        
    """
    # Extraemos las propiedades de saturación en fm^-3 y MeV
    saturacion = plot_saturacion(n_prove, params, plot=False)
    saturation_density = saturacion[0]
    binding_energy = saturacion[1]
    compression_modulus = modulo_compresion(saturation_density, params)
    energy_symmetry_coefficient = coeficiente_simetria(saturation_density, params)
    
    return np.array([saturation_density, binding_energy, compression_modulus, energy_symmetry_coefficient])

#-----------------------------------------------------------------------
# GRAFICAS DE RESULTADOS DE LA ECUACIÓN DE ESTADO
#-----------------------------------------------------------------------

def plot_autoconsistencia(n_prove, params=[A_sigma, A_omega, A_rho, b, c]):
    """
    Grafica la función de autoconsistencia para un rango de densidades bariónicas.
    
    Args:
        n_prove (array-like): Array de densidades bariónicas en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.
        
    Returns:
        None
    """
   
    # Calculamos los valores de x_sigma para las densidades bariónicas dadas
    x_sigma_prove_tilde = np.zeros(len(n_prove))
    x_sigma_prove = np.zeros(len(n_prove))
    for i in range(len(n_prove)):
        x_sigma_prove_tilde[i] = sol_x_sigma_x_nF(n_prove[i], params)
        x_sigma_prove[i] = (1-x_sigma_prove_tilde[i])*m_nuc

    # Mostramos los resultados para x_sigma y x_sigma_tilde en función de n_barion
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].semilogx(n_prove*1e45*m_nuc_MKS*1e-3, x_sigma_prove_tilde) # x_sigma_tilde en función de densidad g/cm^3
    ax[0].set_xlabel(r'$\rho_m$ $[g/cm^3]$')
    ax[0].set_ylabel(r'$\tilde \tilde x_{\sigma}$')
    ax[0].set_title(r'$\tilde x_{\sigma}$ en función de $\rho_m$')
    ax[0].grid()
    ax[1].semilogx(n_prove, x_sigma_prove)
    ax[1].set_xlabel(r'$n_{barion}$ $[fm^{-3}]$')
    ax[1].set_ylabel(r'$x_{\sigma}$')
    ax[1].set_title(r'$x_{\sigma}$ en función de $n_{barion}$')
    ax[1].grid()
    plt.show()
    return None
    
def plot_EoS(rho_P, presiones, energias, n_sirve, rho_0_lambda=m_nuc**4/2, titulo = r'Ecuación de estado'):
    """
    Grafica la ecuación de estado (EoS) y la energía y presión en función de la densidad de masa.
    
    Args:
        rho_P (CubicSpline): Interpolación cúbica de la EoS.
        presiones (array-like): Array de presiones adimenacional.
        energias (array-like): Array de energías adimencional.
        n_sirve (array-like): Array de densidades bariónicas fm^-3.
        rho_0_lambda (float): Densidad de energía en unidades naturales (fm^-4), parámetro de admiencionalización.
        
    Returns:
        None
    """
    
    # Parámetros de conversión
    rho_MKSTocgs = 10
    conv = rho_0_lambda / MeV_to_fm11 * MeVfm_to_Jm * rho_MKSTocgs # Conversión de unidades de presión y energía a cgs (erg/cm^3 y Ba)

    # Creamos la figura con dos subfiguras (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # ----- Subfigura 1: Ecuación de estado con ejes secundarios -----
    ax1.loglog(presiones, energias, "o", label='Puntos de la EoS')
    ax1.loglog(presiones, rho_P(presiones), label='Interpolación')
    ax1.set_xlabel(r'$P/\rho_0$')
    ax1.set_ylabel(r'$\rho/\rho_0$')

    # Agregamos ejes secundarios: eje X superior y eje Y derecho (cgs)
    secax_x = ax1.secondary_xaxis('top', functions=(lambda x: x * conv, lambda x: x / conv))
    secax_x.set_xlabel(r'$P$ [Ba]')
    secax_y = ax1.secondary_yaxis('right', functions=(lambda x: x * conv, lambda x: x / conv))
    secax_y.set_ylabel(r'$\rho$ [erg/cm$^3$]')

    ax1.legend()
    ax1.grid()

    # ----- Subfigura 2: P y energía vs densidad de masa -----
    # Aquí se convierten las unidades de P y energía a cgs usando el mismo factor 'conv'
    x_data = n_sirve * 1e45 * m_nuc_MKS * 1e-3   # Conversión a la densidad de masa (en g/cm^3)

    p_data = presiones * conv  
    e_data = energias  * conv

    ax2.loglog(x_data, p_data, "o", label='Presión [Ba]')
    ax2.loglog(x_data, e_data, "o", label='Energía [erg/cm$^3$]')
    ax2.set_xlabel(r'$\rho_m$ [g/cm$^3$]')
    ax2.set_ylabel(r'$P$ y $\rho$')
    ax2.legend()
    ax2.grid()

    fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()
    return None

def plot_saturacion(n_prove, params=[A_sigma, A_omega, A_rho, b, c], rho_0_lambda=m_nuc**4/2, plot=True):
    """
    Grafica la energía de enlace por nucleón en función de la densidad bariónica y halla la densidad de saturación.
    
    Args:
        n_prove (array-like): Array de densidades bariónicas en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.
        rho_0_lambda (float): Unidades de energía empleadas para adimensionalizar, en naturales (fm^-4).
        plot (bool): Si True, grafica la energía de enlace por nucleón en función de la densidad bariónica.
        
    Returns:
        saturacion (list): Lista con la densidad de saturación en fm^-3 y la energía de enlace por nucleón en MeV (naturales).
    """
    
    energias_prove = np.zeros(len(n_prove)) # Energías de enlace por nucleón en fm^-4 (naturales)
    for i in range(len(n_prove)):
        energias_prove[i], _ = energia_presion(n_prove[i], params)
        energias_prove[i] *= rho_0_lambda # Energías de enlace por nucleón en fm^-4 (naturales)
        
    # Hallamos la densidad de saturación en fm^-3 donde es minima la energía de enlace por nucleón
    minimo = np.argmin(energias_prove/n_prove - m_nuc)
    n_saturacion = n_prove[minimo] # Densidad de saturación en fm^-3
    energia_saturacion = (energias_prove/n_prove - m_nuc)[minimo] / MeV_to_fm11 # Energía de enlace por nucleón en MeV (naturales)
    _, presion_sat = energia_presion(n_saturacion, params)
    presion_sat *= rho_0_lambda

    if plot:
        print(f"La masa efectiva en saturación es: {sol_x_sigma_x_nF(n_saturacion,params):.3f}", "m_nuc")
        print("Densidad de saturación n_saturacion =", format(n_saturacion,".3f"), "1/fm^3 (", format(n_saturacion*1e45*m_nuc_MKS*1e-3,".3e"),"g/cm^3 ) y energia de enlace por nucleon en saturación =", format(energia_saturacion, ".3f"), "MeV y densidad de energia en saturación =", format(energias_prove[minimo]/MeV_to_fm11, ".3f"), "MeV/fm^3")
        print("Presion en la densidad de saturación:", presion_sat/MeV_to_fm11*MeVfm_to_Jm, "Pa")
            
        # Graficamos la energía de enlace por nucleón en función de x_f
        plt.figure(figsize=(8,6))
        plt.plot(n_prove, (energias_prove/n_prove - m_nuc)/MeV_to_fm11, "-o")
        plt.xlabel(r'$n_{barion}$ [fm$^{-3}$]')
        plt.ylabel(r'$\frac{\rho}{n}-m_{nuc}$ [MeV]')
        # plt.ylim(-20, 20)
        # Anotamos el mínimo y su energia con un punto, una flecha y los valores de B/A y n_saturacion con ofset vertical de +2, centrado
        plt.annotate(r'$\left(\frac{B}{A}\right)_{min}=$'+format((energia_saturacion), ".3f")+' MeV'+'\n'+'$n_{sat}$ = '+format(n_saturacion, '.3f')+'$fm^{-3}$', xy=(n_saturacion, energia_saturacion), xytext=(n_saturacion, energia_saturacion+2), arrowprops=dict(arrowstyle='->'), horizontalalignment='center')
        plt.title(r'Energía de enlace por nucleón en función de $n_{barion}$')
        plt.grid()
        plt.show()
        
    return [n_saturacion, energia_saturacion]