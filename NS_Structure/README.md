# NS_Structure
Modelos de materia de estrellas de neutrones, ecuaciones de estado (EoS) y resolución de la ecuación de Tolman–Oppenheimer–Volkoff (TOV). Esta carpeta integra notebooks, scripts, tablas de EoS y resultados precomputados.

## Notebooks principales
- [modelo_sigma-omega.ipynb](modelo_sigma-omega.ipynb)
- [sigma-omega_autointeraccion.ipynb](sigma-omega_autointeraccion.ipynb)
- [sigma-omega-rho.ipynb](sigma-omega-rho.ipynb)
- [neutron-star-matter.ipynb](neutron-star-matter.ipynb)
- [variacion-NS-matter.ipynb](variacion-NS-matter.ipynb)

## Scripts
Scripts para construir EoS, resolver TOV y generar gráficos:
- [scripts/NSMatterEoS.py](scripts/NSMatterEoS.py)
- [scripts/AutointeractuanteEoS.py](scripts/AutointeractuanteEoS.py)
- [scripts/IsospinEoS.py](scripts/IsospinEoS.py)
- [scripts/ResolverTOV.py](scripts/ResolverTOV.py)
- [scripts/Optimizar_IsospinEoS.py](scripts/Optimizar_IsospinEoS.py)
- [scripts/plots_variacion_parametros.py](scripts/plots_variacion_parametros.py)
- [scripts/plot_final_presentation.py](scripts/plot_final_presentation.py)

## Tablas y datos de EoS
- [EoS_tables/](EoS_tables/): tablas y comparaciones, p. ej. [EoS_tables/EoS_crust.txt](EoS_tables/EoS_crust.txt).

## Resultados
- CSV de masas y radios: [results/MasaRadio_NSMatter_rho0_lambda_A_propios.csv](results/MasaRadio_NSMatter_rho0_lambda_A_propios.csv) y archivos similares.
- Mallados y parámetros: [results/EspacioParametros/](results/EspacioParametros/).

## Modelos simples
Notebooks introductorios y de verificación en [Simple_Models/](Simple_Models/):
- [Simple_Models/gas_n-p-e.ipynb](Simple_Models/gas_n-p-e.ipynb)
- [Simple_Models/hidrostaticas.ipynb](Simple_Models/hidrostaticas.ipynb)
- [Simple_Models/neutrones_degenerados.ipynb](Simple_Models/neutrones_degenerados.ipynb)