import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root

#######################################################################
# DEFINICIONES
#######################################################################

# Constantes en unidades MKS
hbar_MKS = 1.0545718e-34 # J·s
c_MKS = 299792458 # m/s
G_MKS = 6.67430e-11 # m³/kg/s²
pi = np.pi
proton_mass = 1.6726219e-27 # kg
neutron_mass = 1.6749275e-27 # kg
electron_mass = 9.1093837e-31 # kg
m_nuc_MKS = (proton_mass + neutron_mass)/2.0 # kg
e_MKS = 1.6021766e-19 # C (coulomb)

# Conversiones útiles (multiplicar al primero para obtener el segundo)
Kg_to_fm11 = c_MKS/hbar_MKS*1e-15 # kg a fm^-1
MeV_to_fm11 = e_MKS/(hbar_MKS*c_MKS*1e9) # MeV a fm^-1
MeVfm_to_Jm  = 1e51*e_MKS # MeV/fm a J/m

# Constantes en unidades naturales
m_nuc = m_nuc_MKS * Kg_to_fm11 # fm^-1
m_e = electron_mass * Kg_to_fm11 # fm^-1

#######################################################################
# Parámetros del modelo (Glendenning)
#######################################################################
A_sigma = 12.684*m_nuc**2
A_omega =  7.148*m_nuc**2
A_rho   =  4.410*m_nuc**2
b       =  5.610e-3
c       = -6.986e-3

#######################################################################
# ECUACIÓN DE ESTADO
#######################################################################

def ecuaciones_autoconsistencia(variables, n_barion, params=[A_sigma, A_omega, A_rho, b, c]):
    """
    Calcula la ecuación de autoconsistencia (ecuación de campo sigma) para el modelosigma-omega-rho
    con autointeracciones del campo escalar y la condición de equilibrio beta en la estrella.

    Args:
        variables (array): Valores para el campo escalar sigma y el momento de fermi del neutron [x_sigma, x_nF].
        n_barion (float): Densidad bariónica en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.

    Returns:
        array: [campo_sigma, equilibrio_beta] (deben ser cero para la solución).
    """
    A_sigma, _, A_rho, b, c = params
    x_sigma, x_nF = variables # Masa efectiva relativa del nucleón y momento de Fermi del neutron
    
    # Momento de Fermi del protón (y electrón, x_pF=x_eF)
    x_pF = ( (3.0*pi**2)*n_barion/m_nuc**3 - x_nF**3)**(1/3) # Momento de Fermi del protón (y electrón)
    
    # Integrales separadas para neutrones y protones
    if x_nF > 0 and x_pF > 0 and x_sigma >= 0:
        # Energías libres para n, p y e con masa efectiva
        raiz_n = np.sqrt(x_nF**2 + x_sigma**2)
        raiz_p = np.sqrt(x_pF**2 + x_sigma**2)
        raiz_e = np.sqrt(x_pF**2 + (m_e/m_nuc)**2)
        
        # Ecuación de campo escalar (=0)
        equilibrio_beta = raiz_n - raiz_p - raiz_e - (A_rho/(6*pi**2)) * (x_pF**3 - x_nF**3)
        
        # Integrales para neutrones y protones de la densidad escalar
        integral_n = 0.5 * x_sigma* (x_nF * raiz_n - x_sigma**2 * np.arctanh(x_nF / raiz_n))
        integral_p = 0.5 * x_sigma * (x_pF * raiz_p - x_sigma**2 * np.arctanh(x_pF / raiz_p))
        integral_total = integral_n + integral_p
        
        # Ecuación de autoconsistencia (=0)
        campo_sigma = (1.0 - x_sigma) - A_sigma * ( integral_total/(pi**2) - b*(1-x_sigma)**2 - c*(1-x_sigma)**3 )
    else:
        # Si los valores son no válidos, devolvemos valores lejos de cero
        campo_sigma = 1e4
        equilibrio_beta = 1e4
    
    # Retornamos las ecuaciones de autoconsistencia y equilibrio beta
    return np.array([campo_sigma, equilibrio_beta])

def sol_x_sigma_x_nF(n_barion, params=[A_sigma, A_omega, A_rho, b, c], x0=[0.5, 0.01], print_solution=False, method='lm'):
    """
    Resuelve las ecuaciones de autoconsistencia para un valor dado de densidad bariónica, encontrando raíces.

    Args:
        n_barion (float): Densidad bariónica en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.
        x0 (list): Valor inicial para la solución [x_sigma, x_nF].
        print_solution (bool): Si es True, imprime detalles de la solución encontrada.
        method (str): Método de optimización para scipy.optimize.root.

    Returns:
        float: Solución para el campo escalar sigma.
    """
    
    # Graficar la superficie de ecuaciones_autoconsistencia para n_barion
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

        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        levels = np.linspace(-1, 1, 21)
        cmap_extended = plt.get_cmap('coolwarm').copy()
        cmap_extended.set_under("#00003F", alpha=0.7)  # Dark Blue for < -1
        cmap_extended.set_over("#510000", alpha=0.7)   # Dark Red for > 1

        # Primer subplot: ecuación de campo del mesón escalar
        c1 = axs[0].contourf(X_sigma, X_nF, Z1, levels=levels, cmap=cmap_extended, extend='both')
        axs[0].contour(X_sigma, X_nF, Z1, levels=[0], colors='#32CD32', linewidths=2)  # Lime Green for Z=0
        fig.colorbar(c1, ax=axs[0], label='Ecuación de autoconsistencia')
        axs[0].set_xlabel('x_sigma')
        axs[0].set_ylabel('x_nF')
        axs[0].set_title(f'Ecuación de campo escalar\nn_barion = {n_barion}')
        line1 = axs[0].axhline(y=x0[1], color='k', linestyle='--', label='Valores Iniciales')
        line2 = axs[0].axvline(x=x0[0], color='k', linestyle='--', label='x_sigma inicial')
        patch_over = mpatches.Patch(color='#510000', alpha=0.7, label='Fuera de rango (>1)')
        patch_under = mpatches.Patch(color='#00003F', alpha=0.7, label='Fuera de rango (<-1)')
        zero_line = Line2D([0], [0], color='#32CD32', lw=2, label='Nivel Z=0')
        axs[0].legend(handles=[line1, zero_line, patch_under, patch_over], fontsize='small')

        # Segundo subplot: ecuación de equilibrio beta
        c2 = axs[1].contourf(X_sigma, X_nF, Z2, levels=levels, cmap=cmap_extended, extend='both')
        axs[1].contour(X_sigma, X_nF, Z2, levels=[0], colors='#32CD32', linewidths=2)  # Lime Green for Z=0
        fig.colorbar(c2, ax=axs[1], label='Ecuación de equilibrio beta')
        axs[1].set_xlabel('x_sigma')
        axs[1].set_ylabel('x_nF')
        axs[1].set_title(f'Ecuación de equilibrio beta\nn_barion = {n_barion}')
        line3 = axs[1].axhline(y=x0[1], color='k', linestyle='--', label='Valores Iniciales')
        line4 = axs[1].axvline(x=x0[0], color='k', linestyle='--', label='x_sigma inicial')
        axs[1].legend(handles=[line3, zero_line, patch_over, patch_under], fontsize='small')

        plt.tight_layout()
        plt.show()

    solution = root(ecuaciones_autoconsistencia, x0, args=(n_barion, params), method=method)
    if not solution.success:
        print("No se encontró solución para n_barion = ", n_barion)
        if print_solution:
            print("Error:", solution.message)
        return [1, 1]
    else:
        if print_solution:
            print("Solución encontrada para n_barion =", n_barion, ":", solution.x)
        return solution.x

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
    
    # Momento de Fermi para protón (y electrón)
    x_pF = (3.0*pi**2*n_barion/m_nuc**3 - x_nF**3)**(1/3)  # Momento de Fermi del protón (y electrón)
   
    # Integrales para neutrones
    if x_nF > 0:
        raiz_n = np.sqrt(x_nF**2 + x_sigma**2)
        termino_arctanh_n = x_sigma**4 * np.arctanh(x_nF / raiz_n)
        integral_energia_n = (x_nF * raiz_n * (2.0 * x_nF**2 + x_sigma**2) - termino_arctanh_n) / 8.0
        integral_presion_n = (x_nF * raiz_n * (2.0 * x_nF**2 - 3.0 * x_sigma**2) + 3.0 * termino_arctanh_n) / 8.0
    else:
        print("Momento de Fermi del neutrón no positivo para n_barion =", n_barion)
        integral_energia_n = 0
        integral_presion_n = 0
    
    # Integrales para protones y electrones
    if x_pF > 0:
        raiz_p = np.sqrt(x_pF**2 + x_sigma**2)
        raiz_e = np.sqrt(x_pF**2 + (m_e/m_nuc)**2)
        termino_arctanh_p = x_sigma**4 * np.arctanh(x_pF / raiz_p)
        termino_arctanh_e = (m_e/m_nuc)**4 * np.arctanh(x_pF / raiz_e)
        integral_energia_p = (x_pF * raiz_p * (2.0 * x_pF**2 + x_sigma**2) - termino_arctanh_p) / 8.0
        integral_energia_e = (x_pF * raiz_e * (2.0 * x_pF**2 + (m_e/m_nuc)**2) - termino_arctanh_e) / 8.0
        integral_presion_p = (x_pF * raiz_p * (2.0 * x_pF**2 - 3.0 * x_sigma**2) + 3.0 * termino_arctanh_p) / 8.0
        integral_presion_e = (x_pF * raiz_e * (2.0 * x_pF**2 - 3.0 * (m_e/m_nuc)**2) + 3.0 * termino_arctanh_e) / 8.0
    else:
        print("Momento de Fermi del protón no positivo para n_barion =", n_barion)
        integral_energia_p = 0
        integral_presion_p = 0
        integral_energia_e = 0
        integral_presion_e = 0
    
    # Términos de los campos
    termino_sigma = (1.0-x_sigma)**2 * (1.0/A_sigma + 2.0/3.0*b*(1.0-x_sigma) + 0.5*c*(1-x_sigma)**2)
    termino_omega_rho = A_omega*(n_barion/m_nuc**3)**2 + 1/(36*pi**4)*A_rho*(x_pF**3 - x_nF**3)**2
   
    
    # Energía y presión totales
    energia = (termino_sigma + termino_omega_rho + 2.0/(pi**2)*(integral_energia_n + integral_energia_p + integral_energia_e))
    presion = (-termino_sigma + termino_omega_rho + 2.0/(3.0*pi**2)*(integral_presion_n + integral_presion_p + integral_presion_e))
    
    return energia, presion

def EoS(n_barions, params=[A_sigma, A_omega, A_rho, b, c], add_crust=False, crust_file_path=None):
    """
    Calcula la ecuación de estado (EoS) para un rango de densidades bariónicas.

    Args:
        n_barions (array-like): Array de densidades bariónicas en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.
        add_crust (bool): Si es True, carga los datos de la corteza y los combina con la EoS del núcleo.
        crust_file_path (str): Ruta al archivo de datos de la corteza (si add_crust es True).

    Returns:
        tuple: CubicSpline de la EoS (P->E), arrays de presiones, energías, densidades y el índice de cambio de signo en la presión.
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
        if presiones[i] < 0 and presiones[i+1] > 0:
            presion_cambio = i+1
            break

    # Core EoS arrays
    rho_0_lambda = m_nuc**4 / 2.0
    ps_core = presiones[presion_cambio:]
    es_core = energias[presion_cambio:]
    ns_core = n_sirve[presion_cambio:]
    # Unir corteza si se solicita
    if add_crust and crust_file_path:
        # Cargar corteza: columnas [ignore, n_crust (fm^-3), rho_mass (g/cm^3), P_cgs (Ba)]
        data = np.loadtxt(crust_file_path)
        n_crust = data[:,1]
        rho_mass = data[:,2]
        P_cgs = data[:,3]
        # Densidad de energía en cgs: E_cgs = rho_mass * c^2
        c_cm = c_MKS * 100  # cm/s
        E_cgs = rho_mass * c_cm**2
        # Conversión natural->cgs para P y E
        rho_MKSTocgs = 10
        conv = rho_0_lambda / MeV_to_fm11 * MeVfm_to_Jm * rho_MKSTocgs
        # Arreglos adimensionales de la corteza
        E_crust = E_cgs / conv
        P_crust = P_cgs / conv
        # Corte en densidad de saturación
        n_sat = 0.161
        mask_cr = n_crust <= n_sat
        mask_co = ns_core >= n_sat
        # Concatenar corteza y núcleo
        n_all = np.concatenate([n_crust[mask_cr], ns_core[mask_co]])
        P_all = np.concatenate([P_crust[mask_cr], ps_core[mask_co]])
        E_all = np.concatenate([E_crust[mask_cr], es_core[mask_co]])
        # Filtrar por el rango de densidades de entrada
        n_min = np.min(n_barions)
        n_max = np.max(n_barions)
        mask_range = (n_all >= n_min) & (n_all <= n_max)
        P_all = P_all[mask_range]
        E_all = E_all[mask_range]
        n_all = n_all[mask_range]
        # Interpolación PchipInterpolator
        try:
            spline = PchipInterpolator(P_all, E_all)
        except Exception as e:
            print("[EoS] Error al crear PchipInterpolator (core+crust):", e)
            mask_unique = np.diff(P_all) > 0
            mask_unique = np.insert(mask_unique, 0, True)
            P_all_f = P_all[mask_unique]
            E_all_f = E_all[mask_unique]
            n_all_f = n_all[mask_unique]
            try:
                spline = PchipInterpolator(P_all_f, E_all_f)
                print("[EoS] PchipInterpolator creado tras filtrar puntos no monótonos/repetidos.")
                return spline, P_all_f, E_all_f, n_all_f, False
            except Exception as e2:
                print("[EoS] Error persistente al crear PchipInterpolator:", e2)
                return None, P_all_f, E_all_f, n_all_f, False
        return spline, P_all, E_all, n_all, False
    # Solo núcleo por defecto
    try:
        spline = PchipInterpolator(ps_core, es_core)
    except Exception as e:
        print("[EoS] Error al crear PchipInterpolator (solo core):", e)
        mask_unique = np.diff(ps_core) > 0
        mask_unique = np.insert(mask_unique, 0, True)
        ps_core_f = ps_core[mask_unique]
        es_core_f = es_core[mask_unique]
        n_core_f = ns_core[mask_unique]
        try:
            spline = PchipInterpolator(ps_core_f, es_core_f)
            print("[EoS] PchipInterpolator creado tras filtrar puntos no monótonos/repetidos.")
            return spline, ps_core_f, es_core_f, n_core_f, presion_cambio
        except Exception as e2:
            print("[EoS] Error persistente al crear PchipInterpolator:", e2)
            return None, ps_core_f, es_core_f, n_core_f, presion_cambio
    return spline, presiones, energias, n_sirve, presion_cambio

def distribucion_especies(n_barions, params=[A_sigma, A_omega, A_rho, b, c]):
    """
    Calcula las densidades de número para neutrones, protones y electrones en función de la densidad bariónica dada.

    Args:
        n_barions (array-like): Array de densidades bariónicas en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] que definen el modelo.

    Returns:
        tuple: Arrays de densidades de número para protones, neutrones y electrones en fm^-3.
    """
    # Recorremos las densidades
    n_neutrones = np.zeros(len(n_barions))
    n_protones = np.zeros(len(n_barions))
    n_electrones = np.zeros(len(n_barions))
    
    for i, n in enumerate(n_barions):
        _, x_nF = sol_x_sigma_x_nF(n, params)
        x_pF = ( (3.0*pi**2)*n/m_nuc**3 - x_nF**3)**(1/3) # Momento de Fermi del protón (y electrón)
        n_neutrones[i] = (1/(3.0*pi**2)) * (x_nF*m_nuc)**3
        n_protones[i] = (1/(3.0*pi**2)) * (x_pF*m_nuc)**3
        n_electrones[i] = n_protones[i] # Neutralidad eléctrica

    return n_protones, n_neutrones, n_electrones


#######################################################################
# GRÁFICAS DE RESULTADOS DE LA ECUACIÓN DE ESTADO
#######################################################################

def plot_autoconsistencia(n_prove, params=[A_sigma, A_omega, A_rho, b, c]):
    """
    Grafica las funciones de autoconsistencia x_sigma y x_nF para un rango de densidades bariónicas.
    
    Se generan dos subplots, uno para x_sigma y otro para x_nF, en donde el eje x principal
    muestra las densidades en fm^-3 y el eje secundario superior muestra las densidades en g/cm^3.
    Se utiliza la relación:
       (g/cm^3) * (1e3/m_nuc_MKS*1e-45) = fm^-3
    Es decir, para convertir de fm^-3 a g/cm^3 se utiliza la función:
        rho_gcm3 = rho_fm3 / (1e3/m_nuc_MKS*1e-45)
    
    Args:
        n_prove (array-like): Array de densidades bariónicas en fm^-3.
        params (list): Lista de parámetros [A_sigma, A_omega, A_rho, b, c] para el modelo.
    
    Returns:
        None
    """
    # Factor de conversión: fm^-3 a g/cm^3
    conv_factor = 1e3 / m_nuc_MKS * 1e-45  # n_fm3 = (g/cm^3)*conv_factor  ==> (g/cm^3) = n_fm3/conv_factor

    # Inicializamos arreglos para las soluciones
    x_sigma_vals = np.zeros(len(n_prove))
    x_nF_vals = np.zeros(len(n_prove))
    
    # Recorremos las densidades de prueba y resolvemos la función de autoconsistencia
    for i, n in enumerate(n_prove):
        sol = sol_x_sigma_x_nF(n, params)
        x_sigma_vals[i] = sol[0]
        x_nF_vals[i] = sol[1]
    
    # Creamos la figura con dos subplots lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    color = "#914098" 

    # Ponemos escala logarítmica en x y añadimos rejilla
    for ax in axs:
        ax.set_xscale('log')
        ax.grid(True, which='major', linestyle='--', alpha=0.8)

    # Primer subplot: x_sigma
    axs[0].plot(n_prove, x_sigma_vals, "o-", color=color)
    axs[0].set_xlabel(r'Densidad bariónica [fm$^{-3}$]')
    axs[0].set_ylabel(r'$x_{\sigma}$')
    axs[0].set_title(r'$x_{\sigma}$ vs densidad (log)')

    secax0 = axs[0].secondary_xaxis(
        'top',
        functions=(lambda x: x/conv_factor, lambda x: x*conv_factor)
    )
    secax0.set_xlabel(r'Densidad [g/cm$^3$]')

    # Segundo subplot: x_nF
    axs[1].plot(n_prove, x_nF_vals, "o-", color=color)
    axs[1].set_xlabel(r'Densidad bariónica [fm$^{-3}$]')
    axs[1].set_ylabel(r'$x_{nF}$')
    axs[1].set_title(r'$x_{nF}$ vs densidad (log)')

    secax1 = axs[1].secondary_xaxis(
        'top',
        functions=(lambda x: x/conv_factor, lambda x: x*conv_factor)
    )
    secax1.set_xlabel(r'Densidad [g/cm$^3$]')

    plt.tight_layout()
    plt.show()
    return None
    
def plot_EoS(rho_P, presiones, energias, n_sirve, rho_0_lambda=m_nuc**4/2, titulo = r'ecuación de estado'):
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

    # Creamos la figura con dos subfiguras (lado a lado)
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