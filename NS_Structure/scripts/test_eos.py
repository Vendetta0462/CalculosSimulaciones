import numpy as np
import matplotlib.pyplot as plt
from NSMatterEoS import EoS, m_nuc, MeV_to_fm11, MeVfm_to_Jm, m_nuc_MKS

# Parámetros y rangos
n_jump = 0.062
# window = 0.155  # +/- alrededor de n_sat
n_barions = np.linspace(0.025, 0.08, 200)
# n_barions = np.linspace(0.01, 0.1, 200)
# params = [320, 200, 7.584e+01, 3.210e-03, -4.086e-03] # MaxMax params
# params = [190, 100, 87.844, 8.11e-3, -4.486e-3] # Menor params
params = [12.684*m_nuc**2, 7.148*m_nuc**2, 4.410*m_nuc**2, 5.610e-3, -6.986e-3] # Base params

# Calcular EoS
s_core, P_core, E_core, n_core, idx_core = EoS(n_barions, params)
# s_core, P_core, E_core, n_core, idx_core = EoS(n_barions)
crust_path = r"c:\Users\nicom\Desktop\CalculosSimulaciones\NS_Structure\EoS_tables\EoS_crust.txt"
s_full, P_full, E_full, n_full, idx_full = EoS(n_barions, params, add_crust=True, crust_file_path=crust_path, n_jump=n_jump)
# s_full, P_full, E_full, n_full, idx_full = EoS(n_barions, add_crust=True, crust_file_path=crust_path)

# Conversión a unidades cgs
conv = (m_nuc**4 / 2.0) / MeV_to_fm11 * MeVfm_to_Jm * 10
n_core = n_core[idx_core:]
P_cgs_core, E_cgs_core = P_core[idx_core:] * conv, E_core[idx_core:] * conv
P_cgs_full, E_cgs_full = P_full * conv, E_full * conv

# Cargar datos de corteza
crust_data = np.loadtxt(crust_path)
mask_crust = (crust_data[:,1] >= n_barions.min()) & (crust_data[:,1] <= n_barions.max())
n_crust_plot = crust_data[:,1][mask_crust]
P_cgs_crust = crust_data[:,3][mask_crust]
E_cgs_crust = crust_data[:,2][mask_crust] * (2.99792458e10)**2

# Calcula el primer valor de densidad del core por encima del jump, y el último valor de densidad de la corteza por debajo del jump
n_core_min = n_core[n_core >= n_jump][0]
n_crust_max = n_crust_plot[n_crust_plot <= n_jump][-1]
print(f'Densidad mínima del núcleo por encima del salto: {n_core_min} fm^-3')
print(f'Densidad máxima de la corteza por debajo del salto: {n_crust_max} fm^-3')

# Configuración de gráficas
plot_loglog = True
fig, (ax_eos, ax_cs2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
plot_fn = ax_eos.loglog if plot_loglog else ax_eos.plot

# Graficar EoS
plot_fn(P_cgs_core, E_cgs_core, '--', label='TRCM', color='green')
# plot_fn(P_cgs_full, E_cgs_full, '.-', label='Núcleo + Corteza')
plot_fn(P_cgs_crust, E_cgs_crust, '-.', label='BPS+BBP', color='blue')

if s_full is not None:
    P_interp = np.logspace(np.log10(P_full.min()), np.log10(P_full.max()), 300) if plot_loglog else np.linspace(P_full.min(), P_full.max(), 300)
    E_interp = s_full(P_interp)
    plot_fn(P_interp * conv, E_interp * conv, color='k', label='Interpolación PCHIP', linewidth=1.5)
    
    # Marcar umbral de saturación
    idx_sat = np.argmin(np.abs(np.where(n_core >= n_jump, n_core, 0) - n_jump))
    print('Densidad umbral core:', n_core[idx_sat])
    idx_sat_crust = np.argmin(np.abs(np.where(n_crust_plot <= n_jump, n_crust_plot, 0) - n_jump))
    print('Densidad umbral crust:', n_crust_plot[idx_sat_crust])
    plot_fn(P_cgs_core[idx_sat], E_cgs_core[idx_sat], 'rx', markersize=10,
            # label=f'{"Umbral" if plot_loglog else "Unión"}: n={n_jump} fm$^{{-3}}$'
            )
    plot_fn(P_cgs_crust[idx_sat_crust], E_cgs_crust[idx_sat_crust], 'rx', markersize=10)
    
    # Velocidad del sonido
    dPdE = np.gradient(P_interp, E_interp)
    (ax_cs2.loglog if plot_loglog else ax_cs2.plot)(P_interp * conv, dPdE, color='k')
    ax_cs2.axhline(1, color='r', linestyle='--', label=r'$c_s^2=1$')
    ax_cs2.set_ylabel(r'$c_s^2$', fontsize=14)
    ax_cs2.legend(loc='center left', fontsize=12)
    ax_cs2.grid(True, linestyle='--', alpha=0.5)
    print('Valor máximo de c_s^2:', dPdE.max())
    if np.any(dPdE > 1): print('¡Advertencia: Se viola la causalidad (c_s^2 > 1) en la interpolación!')

# Configuración final
for ax in [ax_eos, ax_cs2]: ax.set_xlim(left=P_cgs_full.min(), right=P_cgs_full.max())
ax_eos.set_ylim(bottom=E_cgs_full.min(), top=E_cgs_full.max())
ax_cs2.set_xlabel('Presión [Ba]', fontsize=14)
ax_eos.set_ylabel('Densidad de energía [erg/cm$^3$]', fontsize=14)
# ax_eos.set_title('Causalidad de la ecuación de estado completa', fontsize=15)
ax_eos.legend(fontsize=12)
ax_eos.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Checkamos que se cumpla la condición rho >= 3P
if np.any(E_cgs_core < 3 * P_cgs_core):
    # Densidad a la que se viola
    idx_violation = np.where(E_cgs_core < 3 * P_cgs_core)[0][0]
    n_violation = n_core[idx_violation]
    n_mass_violation = n_violation * 1e45 * m_nuc_MKS / 1e3  # g/cm3
    print(f'¡Advertencia: Se viola la condición rho >= 3P en el núcleo a n = {n_violation:.2f} fm^-3 or {n_mass_violation:2.2e} g/cm^3!')