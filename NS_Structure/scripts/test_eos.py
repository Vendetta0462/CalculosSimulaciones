#######################################################################
# Script para comparar EoS core-only vs core+crust
#######################################################################
import numpy as np
import matplotlib.pyplot as plt
from NSMatterEoS import EoS, m_nuc, MeV_to_fm11, MeVfm_to_Jm

#######################################################################
# Rango de densidades bariónicas (fm^-3)
#######################################################################
# n_barions = np.logspace(-2, 0, 200)
# Ventana alrededor de la densidad de saturación
n_sat = 0.161
window = 0.155  # +/- alrededor de n_sat
n_barions = np.linspace(n_sat - window, n_sat + window, 200)

#######################################################################
# EoS sin corteza y con corteza
#######################################################################
s_core, P_core, E_core, n_core, idx_core = EoS(n_barions)
crust_path = r"c:\Users\nicom\Desktop\CalculosSimulaciones\NS_Structure\EoS_tables\EoS_crust.txt"
s_full, P_full, E_full, n_full, idx_full = EoS(n_barions, add_crust=True, crust_file_path=crust_path)

#######################################################################
# Graficar en unidades cgs alrededor de la densidad de saturación
#######################################################################
rho_0_lambda = m_nuc**4 / 2.0
rho_MKSTocgs = 10
conv = rho_0_lambda / MeV_to_fm11 * MeVfm_to_Jm * rho_MKSTocgs
P_cgs_core = P_core * conv
E_cgs_core = E_core * conv
P_cgs_full = P_full * conv
E_cgs_full = E_full * conv

#######################################################################
# Graficar también el crust solo
#######################################################################
crust_data = np.loadtxt(crust_path)
n_crust = crust_data[:,1]
rho_mass = crust_data[:,2]
P_cgs_crust = crust_data[:,3]
c_cm = 2.99792458e10  # cm/s
E_cgs_crust = rho_mass * c_cm**2
mask_crust = (n_crust >= n_barions.min()) & (n_crust <= n_barions.max())
P_cgs_crust = P_cgs_crust[mask_crust]
E_cgs_crust = E_cgs_crust[mask_crust]
n_crust_plot = n_crust[mask_crust]

#######################################################################
# Graficar EoS y velocidad del sonido cuadrado en subplots
#######################################################################
plot_loglog = True  # Cambia a False para escala lineal
fig, (ax_eos, ax_cs2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
# EoS arriba
if plot_loglog:
    ax_eos.loglog(P_cgs_core, E_cgs_core, '.-', label='Core only')
    ax_eos.loglog(P_cgs_full, E_cgs_full, '.-', label='Core + Crust')
    ax_eos.loglog(P_cgs_crust, E_cgs_crust, '.-', label='Crust only')
else:
    ax_eos.plot(P_cgs_core, E_cgs_core, '.-', label='Core only')
    ax_eos.plot(P_cgs_full, E_cgs_full, '.-', label='Core + Crust')
    ax_eos.plot(P_cgs_crust, E_cgs_crust, '.-', label='Crust only')
if s_full is not None:
    P_interp = np.logspace(np.log10(P_full.min()), np.log10(P_full.max()), 300) if plot_loglog else np.linspace(P_full.min(), P_full.max(), 300)
    E_interp = s_full(P_interp)
    if plot_loglog:
        ax_eos.loglog(P_interp * conv, E_interp * conv, '--', color='k', label='Interpolación core+crust')
    else:
        ax_eos.plot(P_interp * conv, E_interp * conv, '--', color='k', label='Interpolación core+crust')
    # Marca en densidad de saturación
    idx_sat = np.argmin(np.abs(n_barions - n_sat))
    if plot_loglog:
        ax_eos.loglog(P_cgs_core[idx_sat], E_cgs_core[idx_sat], 'x', markersize=10, label='Saturation')
    else:
        ax_eos.plot(P_cgs_core[idx_sat], E_cgs_core[idx_sat], 'x', markersize=10, label='Saturation')
    # Velocidad del sonido cuadrado abajo
    dPdE = np.gradient(P_interp, E_interp)
    if plot_loglog:
        ax_cs2.loglog(P_interp * conv, dPdE, label=r'$c_s^2 = dP/dE$')
    else:
        ax_cs2.plot(P_interp * conv, dPdE, label=r'$c_s^2 = dP/dE$')
    ax_cs2.axhline(1, color='r', linestyle='--', label=r'Causalidad ($c_s^2=1$)')
    ax_cs2.set_ylabel(r'$c_s^2$')
    ax_cs2.legend()
    ax_cs2.grid(True, which='both', linestyle='--', alpha=0.5)
    print('Valor máximo de c_s^2:', dPdE.max())
    if np.any(dPdE > 1):
        print('¡Advertencia: Se viola la causalidad (c_s^2 > 1) en la interpolación!')
ax_cs2.set_xlabel('Presión [Ba]')
ax_eos.set_ylabel('Densidad de energía [erg/cm^3]')
ax_eos.set_title('Comparación EoS cerca de la densidad de saturación')
ax_eos.legend()
ax_eos.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
