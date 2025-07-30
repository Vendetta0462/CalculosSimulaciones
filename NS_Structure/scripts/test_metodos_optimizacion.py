"""
Script para evaluar diferentes métodos de optimización de lmfit
y comparar resultados (chi2) y tiempos de ejecución.
Prueba también el bloqueo de un parámetro a la vez.
"""
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from Optimizar_IsospinEoS import optimizar_parametros, crear_propiedades_objetivo, PARAMETER_NAMES


def evaluar_metodos():
    # Propiedades objetivo base
    props = crear_propiedades_objetivo()
    # Métodos de lmfit a probar
    methods = [
        'leastsq',
        'nelder',
        'differential_evolution',
        'lbfgsb',
        'powell',
        'cg'
    ]
    # Configuraciones: todos libres + uno fijo a la vez
    configuraciones = [([], 'all_free')]
    configuraciones += [([p], f'fixed_{p}') for p in PARAMETER_NAMES]

    resultados = []
    print("Iniciando pruebas de optimización...")

    for metodo in methods:
        for fixed_params, label in configuraciones:
            print(f"Método: {metodo}, Config: {label}")
            start = time.perf_counter()
            res = optimizar_parametros(
                props,
                metodo=metodo,
                verbose=False,
                fixed_params=fixed_params
            )
            elapsed = time.perf_counter() - start
            chi2 = res.get('chi2_final', None)
            success = res.get('exito', False)
            print(f"  Chi2: {chi2}, Tiempo: {elapsed:.2f}s, Éxito: {success}\n")
            resultados.append({
                'method': metodo,
                'config': label,
                'chi2': chi2,
                'time_s': round(elapsed, 4),
                'success': success
            })

    # Guardar resultados en CSV
    archivo = 'resultados_metodos.csv'
    with open(archivo, 'w', newline='') as csvfile:
        campos = ['method', 'config', 'chi2', 'time_s', 'success']
        writer = csv.DictWriter(csvfile, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"Pruebas completadas. Resultados en: {archivo}")


def resumen_y_visualizacion(csv_file='resultados_metodos.csv'):
    """
    Carga CSV de resultados, muestra resumen y gráficos de chi2 vs tiempo.
    """
    df = pd.read_csv(csv_file)
    # Filtrar sólo exitosos
    df_ok = df[df['success']]
    # Mostrar mejores chi2
    print("\nTop 5 mejores ajustes (menor chi2):")
    print(df_ok.nsmallest(5, 'chi2')[['method','config','chi2','time_s']])
    # Mostrar más rápidos
    print("\nTop 5 optimizaciones más rápidas:")
    print(df_ok.nsmallest(5, 'time_s')[['method','config','chi2','time_s']])
    # Gráfico dispersión chi2 vs tiempo
    plt.figure(figsize=(8,6))
    for cfg in df_ok['config'].unique():
        sub = df_ok[df_ok['config']==cfg]
        plt.scatter(sub['time_s'], sub['chi2'], label=cfg)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Chi²')
    plt.title('Chi² vs Tiempo por configuración')
    plt.legend(title='Config', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Gráfico de barras: chi2 por método y configuración
    pivot = df_ok.pivot(index='method', columns='config', values='chi2')
    pivot.plot(kind='bar', figsize=(10,6))
    plt.ylabel('Chi²')
    plt.title('Chi² por método y configuración')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Deseas realizar pruebas (1) o visualizar resultados (2)?", end=" ")
    opcion = input().strip().lower()
    if opcion == "1":
        evaluar_metodos()
    elif opcion == "2":
        resumen_y_visualizacion('resultados_metodos.csv')
    else:
        print("Opción no válida.")
