# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
import IsospinEoS as isoEoS

#-----------------------------------------------------------------------
# CONSTANTES Y CONFIGURACIÓN
#-----------------------------------------------------------------------

# Valores base de las propiedades de saturación nuclear
BASE_VALUES = {
    'n_sat': 0.161,      # fm^-3, densidad de saturación
    'B_A_sat': -16.24,   # MeV, energía de enlace por nucleón
    'K_mod': 230.0,      # MeV, módulo de compresión
    'a_sym': 31.6        # MeV, coeficiente de energía de simetría
}

# Rangos razonables para los parámetros del modelo (basados en literatura)
PARAMETER_BOUNDS = {
    'A_sigma': (5.0, 20.0),     # en unidades de m_nuc^2
    'A_omega': (1.0, 15.0),     # en unidades de m_nuc^2
    'A_rho': (0.0, 10.0),       # en unidades de m_nuc^2
    'b': (-0.01, 0.01),         # adimensional
    'c': (-0.01, 0.01)          # adimensional
}

# Valores iniciales por defecto (valores de Glendenning, los A_campo en m_nuc^2)
DEFAULT_INITIAL_PARAMS = {
    'A_sigma': 12.684,  # m_nuc^2
    'A_omega': 7.148,   # m_nuc^2
    'A_rho': 4.410,     # m_nuc^2
    'b': 5.610e-3,      # adimensional
    'c': -6.986e-3      # adimensional
}

# Nombres de los parámetros del modelo
PARAMETER_NAMES = ['A_sigma', 'A_omega', 'A_rho', 'b', 'c']

#-----------------------------------------------------------------------
# FUNCIONES DE OPTIMIZACIÓN
#-----------------------------------------------------------------------

def objetivo_residuos(params, propiedades_objetivo, n_range, pesos=None, verbose=True):
    """
    Función objetivo que calcula los residuos entre las propiedades calculadas 
    y las propiedades objetivo.
    
    Args:
        params (array): Parámetros del modelo [A_sigma, A_omega, A_rho, b, c].
        propiedades_objetivo (dict): Diccionario con las propiedades objetivo
                                   {'n_sat': valor, 'B_A_sat': valor, 'K_mod': valor, 'a_sym': valor}
        n_range (array): Rango de densidades para calcular las propiedades.
        pesos (array, optional): Pesos para cada propiedad en la función objetivo.
        verbose (bool): Si mostrar información del proceso de cálculo.
        
    Returns:
        array: Array de residuos normalizados.
    """
    # Extraer parámetros de lmfit.Parameters o array
    if isinstance(params, Parameters):
        vals = params.valuesdict()
        A_sigma, A_omega, A_rho = vals['A_sigma'], vals['A_omega'], vals['A_rho']
        b, c = vals['b'], vals['c']
    else:
        # lista, tuple o ndarray: convertir a float
        A_sigma, A_omega, A_rho, b, c = [float(v) for v in params]
    t = 0.0  # Para materia simétrica
    
    # Verificar que los parámetros están dentro de rangos físicos razonables
    if (A_sigma <= 0 or A_omega <= 0 or A_rho <= 0):
        return np.array([1e6, 1e6, 1e6, 1e6])  # Penalización alta
    
    try:
        # Calcular las propiedades del modelo con los parámetros dados
        params_modelo = [A_sigma, A_omega, A_rho, b, c, t]
        propiedades_calculadas = isoEoS.calculate_properties(n_range, params_modelo, verbose=verbose)

        n_sat_calc, B_A_sat_calc, K_mod_calc, a_sym_calc = propiedades_calculadas
        
        # Extraer valores objetivo
        n_sat_obj = propiedades_objetivo['n_sat']
        B_A_sat_obj = propiedades_objetivo['B_A_sat']
        K_mod_obj = propiedades_objetivo['K_mod']
        a_sym_obj = propiedades_objetivo['a_sym']
        
        # Calcular residuos normalizados
        residuo_n_sat = (n_sat_calc - n_sat_obj) / n_sat_obj
        residuo_B_A = (B_A_sat_calc - B_A_sat_obj) / abs(B_A_sat_obj)
        residuo_K_mod = (K_mod_calc - K_mod_obj) / K_mod_obj
        residuo_a_sym = (a_sym_calc - a_sym_obj) / a_sym_obj
        
        # Array de residuos principales
        residuos = np.array([residuo_n_sat, residuo_B_A, residuo_K_mod, residuo_a_sym])
        # Aplicar pesos si se proporcionan (solo a los 4 primeros)
        if pesos is not None:
            residuos[:4] *= pesos
        # Añadir norma Euclidiana como quinto componente
        norma_res = np.linalg.norm(residuos)
        residuos = np.append(residuos, norma_res)
        return residuos
        
    except Exception as e:
        print(f"Error en el cálculo de propiedades: {e}")
        return np.array([1e6, 1e6, 1e6, 1e6])  # Penalización alta

def optimizar_parametros(propiedades_objetivo, metodo='leastsq', 
                        params_iniciales=None, n_range=None, pesos=None, 
                        bounds=None, verbose=True, **kwargs):
    """
    Optimiza los parámetros del modelo sigma-omega-rho para ajustar
    las propiedades de saturación especificadas.
    
    Args:
        propiedades_objetivo (dict): Diccionario con las propiedades objetivo
                                   {'n_sat': valor, 'B_A_sat': valor, 'K_mod': valor, 'a_sym': valor}
        metodo (str): Método de optimización ('leastsq', 'nelder', 'differential_evolution')
        params_iniciales (array, optional): Parámetros iniciales [A_sigma, A_omega, A_rho, b, c]
        n_range (array, optional): Rango de densidades para calcular las propiedades
        pesos (array, optional): Pesos para cada propiedad [w_nsat, w_BA, w_K, w_asym]
        bounds (list, optional): Lista de tuplas (min, max) para cada parámetro
        verbose (bool): Si mostrar información del proceso de optimización
        **kwargs: Argumentos adicionales para el optimizador lmfit.minimize()
        
    Returns:
        dict: Diccionario con los resultados de la optimización
    """
    
    # Configurar parámetros iniciales
    if params_iniciales is None:
        m2 = isoEoS.m_nuc**2
        params_iniciales = [DEFAULT_INITIAL_PARAMS[k]*m2 for k in ['A_sigma', 'A_omega', 'A_rho']
                            ] + [DEFAULT_INITIAL_PARAMS[k] for k in ['b', 'c']]
    
    # Configurar bounds
    if bounds is None:
        m2 = isoEoS.m_nuc**2
        bounds = [(lo*m2, hi*m2) for lo, hi in (PARAMETER_BOUNDS[k] for k in ("A_sigma", "A_omega", "A_rho"))
                 ] + [PARAMETER_BOUNDS[k] for k in ("b", "c")]
    
    # Configurar rango de densidades
    if n_range is None:
        n_range = np.logspace(-3, -0.1, 100) # De 0.001 a 0.794 fm^-3

    if verbose:
        print(f"Iniciando optimización con método: {metodo}")
        print("Parámetros iniciales:")
        for param, valor in zip(PARAMETER_NAMES, params_iniciales):
            print(f"  {param}: {valor:.4f}")
        print()

    # Configurar parámetros de lmfit
    params = Parameters()
    for i, name in enumerate(PARAMETER_NAMES):
        params.add(name, value=params_iniciales[i], min=bounds[i][0], max=bounds[i][1])

    # Realizar optimización con lmfit
    resultado = minimize(
        objetivo_residuos,
        params,
        args=(propiedades_objetivo,),
        kws={'n_range': n_range, 'pesos': pesos, 'verbose': verbose},
        method=metodo,
        **kwargs
    )
    # Extraer resultados optimizados desde lmfit
    params_optimizados = [resultado.params[name].value for name in PARAMETER_NAMES]
    chi2_final = getattr(resultado, 'chisqr', None)
    exito = getattr(resultado, 'success', True)
    mensaje = getattr(resultado, 'errmsg', '') or getattr(resultado, 'message', '')
    
    # Calcular las propiedades finales con los parámetros optimizados
    try:
        A_sigma, A_omega, A_rho, b, c = params_optimizados
        params_modelo = [A_sigma, A_omega, A_rho, b, c, 0.0]  # t=0
        propiedades_finales = isoEoS.calculate_properties(n_range, params_modelo, verbose=verbose)
        n_sat_final, B_A_sat_final, K_mod_final, a_sym_final = propiedades_finales
        
        propiedades_calculadas = {
            'n_sat': n_sat_final,
            'B_A_sat': B_A_sat_final,
            'K_mod': K_mod_final,
            'a_sym': a_sym_final
        }
        
    except Exception as e:
        print(f"Error calculando propiedades finales: {e}")
        propiedades_calculadas = None
    
    # Crear diccionario de resultados
    resultados = {
        'parametros_optimizados': {name: params_optimizados[i] for i, name in enumerate(PARAMETER_NAMES)},
        'propiedades_objetivo': propiedades_objetivo,
        'propiedades_calculadas': propiedades_calculadas,
        'chi2_final': chi2_final,
        'exito': exito,
        'mensaje': mensaje,
        'metodo_usado': metodo,
        'resultado_completo': resultado
    }
    
    if verbose and propiedades_calculadas:
        print("Optimización completada!")
        print(f"Éxito: {exito}")
        print(f"Chi² final: {chi2_final:.6f}")
        print("\nParámetros optimizados:")
        for param, valor in resultados['parametros_optimizados'].items():
            print(f"  {param}: {valor:.6f}")
        print("\nComparación de propiedades:")
        print(f"{'Propiedad':<10} {'Objetivo':<12} {'Calculado':<12} {'Error rel (%)':<12}")
        print("-" * 50)
        for prop in ['n_sat', 'B_A_sat', 'K_mod', 'a_sym']:
            obj = propiedades_objetivo[prop]
            calc = propiedades_calculadas[prop]
            error_rel = 100 * abs(calc - obj) / abs(obj)
            print(f"{prop:<10} {obj:<12.4f} {calc:<12.4f} {error_rel:<12.2f}")
    
    return resultados

def estudiar_parametros_vs_propiedades(propiedades_base, variaciones_propiedades, 
                                     nombres_propiedades=None, metodo='leastsq',
                                     n_range=None, **kwargs):
    """
    Estudia cómo varían los parámetros optimizados cuando se cambian las propiedades objetivo.
    
    Args:
        propiedades_base (dict): Propiedades base de referencia
        variaciones_propiedades (dict): Diccionario con arrays de variaciones para cada propiedad
                                      {'n_sat': array, 'B_A_sat': array, etc.}
        nombres_propiedades (list, optional): Lista de nombres de propiedades a variar
        metodo (str): Método de optimización a usar
        n_range (array, optional): Rango de densidades
        **kwargs: Argumentos adicionales para el optimizador
        
    Returns:
        dict: Resultados del estudio paramétrico
    """
    
    if nombres_propiedades is None:
        nombres_propiedades = list(variaciones_propiedades.keys())
    
    resultados_estudio = {}
    
    for nombre_prop in nombres_propiedades:
        print(f"\nEstudiando variación de {nombre_prop}...")
        
        valores_prop = variaciones_propiedades[nombre_prop]
        resultados_prop = []
        
        for valor in valores_prop:
            # Crear propiedades objetivo con la variación
            props_objetivo = propiedades_base.copy()
            props_objetivo[nombre_prop] = valor
            
            # Optimizar parámetros
            resultado = optimizar_parametros(
                props_objetivo, 
                metodo=metodo, 
                n_range=n_range, 
                verbose=False,
                **kwargs
            )
            
            if resultado['exito']:
                resultados_prop.append({
                    'valor_propiedad': valor,
                    'parametros': resultado['parametros_optimizados'],
                    'propiedades_calculadas': resultado['propiedades_calculadas'],
                    'chi2': resultado['chi2_final']
                })
        
        resultados_estudio[nombre_prop] = resultados_prop
    
    return resultados_estudio

#-----------------------------------------------------------------------
# FUNCIONES DE VISUALIZACIÓN
#-----------------------------------------------------------------------

def plot_convergencia_optimizacion(resultados, save_fig=False, filename=None):
    """
    Grafica la convergencia de la optimización y comparación de propiedades.
    
    Args:
        resultados (dict): Resultados de la optimización
        save_fig (bool): Si guardar la figura
        filename (str): Nombre del archivo para guardar
    """
    
    if not resultados['propiedades_calculadas']:
        print("No hay propiedades calculadas para graficar")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Comparación propiedades objetivo vs calculadas
    propiedades = ['n_sat', 'B_A_sat', 'K_mod', 'a_sym']
    # Aplicar signo invertido para energía de enlace (B_A_sat es negativa)
    valores_obj = [(-resultados['propiedades_objetivo'][prop] if prop=='B_A_sat' else resultados['propiedades_objetivo'][prop]) for prop in propiedades]
    valores_calc = [(-resultados['propiedades_calculadas'][prop] if prop=='B_A_sat' else resultados['propiedades_calculadas'][prop]) for prop in propiedades]
    
    x_pos = np.arange(len(propiedades))
    width = 0.35
    
    ax1.bar(x_pos - width/2, valores_obj, width, label='Objetivo', alpha=0.8)
    ax1.bar(x_pos + width/2, valores_calc, width, label='Calculado', alpha=0.8)
    
    ax1.set_xlabel('Propiedades')
    ax1.set_ylabel('Valores (escala log)')
    ax1.set_title('Comparación: Objetivo vs Calculado')
    ax1.set_yscale('log')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['$n_{sat}$', '$-B/A_{sat}$', '$K_{mod}$', '$a_{sym}$'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Errores relativos
    errores_rel = []
    for prop in propiedades:
        obj = resultados['propiedades_objetivo'][prop]
        calc = resultados['propiedades_calculadas'][prop]
        error_rel = 100 * abs(calc - obj) / abs(obj)
        errores_rel.append(error_rel)
    
    ax2.bar(x_pos, errores_rel, color='red', alpha=0.7)
    ax2.set_xlabel('Propiedades')
    ax2.set_ylabel('Error relativo (%)')
    ax2.set_title('Errores relativos de ajuste')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['$n_{sat}$', '$B/A_{sat}$', '$K_{mod}$', '$a_{sym}$'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_parametros_vs_propiedad(resultados_estudio, nombre_propiedad, save_fig=False, filename=None):
    """
    Grafica cómo varían todos los parámetros en función de una propiedad específica.
    
    Args:
        resultados_estudio (dict): Resultados del estudio paramétrico
        nombre_propiedad (str): Nombre de la propiedad estudiada
        save_fig (bool): Si guardar la figura
        filename (str): Nombre del archivo para guardar
    """
    
    if nombre_propiedad not in resultados_estudio:
        print(f"No hay datos para la propiedad {nombre_propiedad}")
        return
    
    datos = resultados_estudio[nombre_propiedad]
    
    # Extraer datos
    valores_propiedad = [d['valor_propiedad'] for d in datos]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(PARAMETER_NAMES):
        valores_param = [d['parametros'][param] for d in datos]
        
        axes[i].plot(valores_propiedad, valores_param, 'o-', linewidth=2, markersize=6)
        axes[i].set_xlabel(f'{nombre_propiedad}')
        axes[i].set_ylabel(f'{param}')
        axes[i].set_title(f'{param} vs {nombre_propiedad}')
        axes[i].grid(True, alpha=0.3)
    
    # Ocultar el último subplot si hay un número impar de parámetros
    if len(PARAMETER_NAMES) % 2 != 0:
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    
    if save_fig and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

#-----------------------------------------------------------------------
# FUNCIONES AUXILIARES
#-----------------------------------------------------------------------

def crear_propiedades_objetivo(n_sat=None, B_A_sat=None, K_mod=None, a_sym=None):
    """
    Crea un diccionario de propiedades objetivo, usando valores experimentales por defecto.
    
    Args:
        n_sat (float, optional): Densidad de saturación en fm^-3
        B_A_sat (float, optional): Energía de enlace por nucleón en MeV
        K_mod (float, optional): Módulo de compresión en MeV  
        a_sym (float, optional): Coeficiente de energía de simetría en MeV
        
    Returns:
        dict: Diccionario con las propiedades objetivo
    """
    
    propiedades = BASE_VALUES.copy()
    
    if n_sat is not None:
        propiedades['n_sat'] = n_sat
    if B_A_sat is not None:
        propiedades['B_A_sat'] = B_A_sat
    if K_mod is not None:
        propiedades['K_mod'] = K_mod
    if a_sym is not None:
        propiedades['a_sym'] = a_sym
        
    return propiedades

def validar_parametros(params):
    """
    Valida que los parámetros estén dentro de rangos físicos razonables.
    
    Args:
        params (array): Parámetros [A_sigma, A_omega, A_rho, b, c]
        
    Returns:
        bool: True si los parámetros son válidos
    """
    
    A_sigma, A_omega, A_rho, b, c = params
    
    # Convertir a unidades adimensionales para comparar con bounds
    A_sigma_adim = A_sigma / isoEoS.m_nuc**2
    A_omega_adim = A_omega / isoEoS.m_nuc**2  
    A_rho_adim = A_rho / isoEoS.m_nuc**2
    
    validacion = (
        PARAMETER_BOUNDS['A_sigma'][0] <= A_sigma_adim <= PARAMETER_BOUNDS['A_sigma'][1] and
        PARAMETER_BOUNDS['A_omega'][0] <= A_omega_adim <= PARAMETER_BOUNDS['A_omega'][1] and
        PARAMETER_BOUNDS['A_rho'][0] <= A_rho_adim <= PARAMETER_BOUNDS['A_rho'][1] and
        PARAMETER_BOUNDS['b'][0] <= b <= PARAMETER_BOUNDS['b'][1] and
        PARAMETER_BOUNDS['c'][0] <= c <= PARAMETER_BOUNDS['c'][1]
    )
    
    return validacion

def ejemplo_uso(metodo):
    """
    Función de ejemplo que muestra cómo usar las funciones de optimización.
    """
    
    print("=== EJEMPLO DE USO DEL OPTIMIZADOR ===")
    
    # 1. Crear propiedades objetivo usando valores experimentales
    props_objetivo = crear_propiedades_objetivo()

    # 2. Optimizar parámetros
    print("\n=== OPTIMIZACIÓN ===")
    resultados = optimizar_parametros(
        props_objetivo, 
        metodo=metodo,
        verbose=True
    )
    
    # 3. Graficar resultados
    print("\n=== VISUALIZACIÓN ===")
    plot_convergencia_optimizacion(resultados)
    
    return resultados

if __name__ == "__main__":
    # Ejecutar ejemplo si el script se ejecuta directamente
    metodo = input("Ingrese el método de optimización (leastsq, nelder, differential_evolution, ...): ")
    if not metodo:
        print("Usando 'leastsq' por defecto.")
        metodo = 'leastsq'
    ejemplo_uso(metodo)
