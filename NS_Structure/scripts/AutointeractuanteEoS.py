# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

# Definimos las constantes necesarias en MKS
hbar_MKS = 1.0545718e-34 # J s
c_MKS = 299792458 # m/s
G_MKS = 6.67430e-11 # m^3/kg/s^2
pi = np.pi
m_nuc_MKS = 1.6726219e-27 # kg

# Definimos las constantes necesarias en unidades geometrizadas
hbar = hbar_MKS * (G_MKS/c_MKS**3) # m^2
m_nuc = m_nuc_MKS * (G_MKS/c_MKS**2) # m

def autoconsistencia(x_sigma, n_barion, params=[330.263, 249.547, 1, 1]):
    """
    Calcula la ecuación de autoconsistencia para el modelo sigma-omega con autointeracciones del campo escalar.

    Args:
        x_sigma (float): Valor para el campo escalar sigma.
        n_barion (float): Densidad bariónica.
        params (list): Lista de parámetros [A_sigma, A_omega, b_, c_] que definen el modelo.

    Returns:
        float: Valor de la función de autoconsistencia que debe ser cero para encontrar la solución.
    """
    A_sigma, A_omega, b_, c_ = params
    x_f = (1.0/m_nuc)*hbar*(3.0*pi**2*n_barion/2.0)**(1/3)
    raiz = np.sqrt(x_f**2+x_sigma**2)
    integral = x_sigma*(x_f*raiz-x_sigma**2*np.arctanh(x_f/raiz))
    return (1.0 - x_sigma) - A_sigma*(integral/(pi**2)-b_*(1-x_sigma)**2-c_*(1-x_sigma)**3)

def sol_x_sigma(n_barion, params=[330.263, 249.547, 1, 1]):
    """
    Resuelve la ecuación de autoconsistencia para un valor dado de densidad bariónica, encontrando raíces.

    Args:
        n_barion (float): Densidad bariónica.
        params (list): Lista de parámetros [A_sigma, A_omega, b_, c_] que definen el modelo.

    Returns:
        float: Solución para el campo escalar sigma.
    """
    solution = fsolve(autoconsistencia, 0.5, args=(n_barion, params), full_output=True)
    if solution[2] != 1:
        print("No se encontró solución para n_barion = ", n_barion)
        return 0
    else:
        return solution[0][0]

def energia_presion(n_barion, params=[330.263, 249.547, 1, 1]):
    """
    Calcula la energía y la presión para una densidad bariónica dada.

    Args:
        n_barion (float): Densidad bariónica.
        params (list): Lista de parámetros [A_sigma, A_omega, b_, c_] que definen el modelo.

    Returns:
        tuple: Energía y presión calculadas.
    """
    A_sigma, A_omega, b_, c_ = params
    x_sigma = sol_x_sigma(n_barion, params)
    x_f = (1.0/m_nuc)*hbar*(3.0*pi**2*n_barion/2.0)**(1/3)
    lambda_ = m_nuc**4/hbar**3
    raiz = np.sqrt(x_f**2+x_sigma**2)
    termino_arctanh = x_sigma**4*np.arctanh(x_f/raiz)
    integral_energia = (x_f*raiz*(2.0*x_f**2+x_sigma**2)-termino_arctanh)/8.0
    integral_presion = (x_f*raiz*(2.0*x_f**2-3.0*x_sigma**2)+3.0*termino_arctanh)/8.0
    termino_sigma = (1.0-x_sigma)**2*(1.0/A_sigma + 2.0/3.0*b_*(1.0-x_sigma) + 0.5*c_*(1-x_sigma)**2)
    termino_omega = 4.0*A_omega*x_f**6/(9.0*pi**4)
    energia = (termino_sigma + termino_omega + 4.0/(pi**2)*integral_energia)
    presion = (-termino_sigma + termino_omega + 4.0/(3.0*pi**2)*integral_presion)
    return energia, presion

def EoS(n_barions, params=[330.263, 249.547, 1, 1]):
    """
    Calcula la ecuación de estado (EoS) para un rango de densidades bariónicas.

    Args:
        n_barions (array-like): Array de densidades bariónicas.
        params (list): Lista de parámetros [A_sigma, A_omega, b_, c_] que definen el modelo.

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