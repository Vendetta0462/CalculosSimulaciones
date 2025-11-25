
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import sys
import os

# Add current directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import NSMatterEoS as NSEoS
import ResolverTOV as tov

# ==========================================
# Constants and Units
# ==========================================
hbar_MKS = 1.0545718e-34 # J s
c_MKS = 299792458 # m/s
G_MKS = 6.67430e-11 # m^3/kg/s^2
pi = np.pi
proton_mass = 1.6726219e-27 # kg
neutron_mass = 1.6749275e-27 # kg
m_nuc_MKS = (proton_mass + neutron_mass)/2.0 # kg
e_MKS = 1.6021766e-19 # C

# Conversions
Kg_to_fm11 = c_MKS/hbar_MKS*1e-15 # kg to fm^-1
MeV_to_fm11 = e_MKS/(hbar_MKS*c_MKS*1e9) # MeV to fm^-1
MeVfm_to_Jm  = 1e51*e_MKS # MeV/fm to J/m

# Natural units
m_nuc = m_nuc_MKS * Kg_to_fm11 # fm^-1

# Scaling density (rho_0_lambda) in MKS (J/m^3)
rho_nat_to_MKS = 1.0 / MeV_to_fm11 * MeVfm_to_Jm
rho_0_lambda_nat = m_nuc**4 / 2.0
rho_0_lambda_MKS = rho_0_lambda_nat * rho_nat_to_MKS

# ==========================================
# EoS Setup
# ==========================================
# Parameters (Glendenning)
A_sigma = 12.684*m_nuc**2
A_omega =  7.148*m_nuc**2
A_rho   =  4.410*m_nuc**2
b       =  5.610e-3
c       = -6.986e-3
params = [A_sigma, A_omega, A_rho, b, c]

# Generate EoS
print("Generating EoS...")
# Range of densities
densidad_masa_max = 1e17*1e3*(1e-45/m_nuc_MKS) # 1e17 g/cm^3 to fm^-3
densidad_masa_min = 5e9*1e3*(1e-45/m_nuc_MKS) # 5e9 g/cm^3 to fm^-3
n_barions = np.logspace(np.log10(densidad_masa_min), np.log10(densidad_masa_max), 500)

# Calculate EoS using NSMatterEoS
# We use add_crust=False to keep it simple as per request "utilizando la EoS #file:NSMatterEoS.py con los parámetros por defecto"
# But usually for TOV we need crust. The user said "con los parámetros por defecto".
# Default add_crust is False in the function definition, but in the notebook it is True.
# I will use False to avoid file path issues with crust file, unless I can find it.
# The crust file is at 'EoS_tables/EoS_crust.txt'.
# I'll try to find it.
crust_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'EoS_tables', 'EoS_crust.txt')
add_crust = False
if os.path.exists(crust_path):
    add_crust = True
    print(f"Crust file found at {crust_path}")
else:
    print("Crust file not found, using core EoS only.")

rho_P_func, presiones, energias, n_sirve, presion_cambio = NSEoS.EoS(n_barions, params, add_crust=add_crust, crust_file_path=crust_path)

# Create interpolators
# rho_P_func is P -> rho (dimensionless)
# We need rho -> P to find P_central
P_from_rho = PchipInterpolator(energias, presiones)
Energy_from_n = PchipInterpolator(n_sirve, energias)
epsilon_limit_dim = energias[0]

# ==========================================
# TOV Integration
# ==========================================
# Central density: 10^15 g/cm^3
rho_central_cgs = 1e15 # g/cm^3

# Convert to baryon density (fm^-3)
# n (fm^-3) = rho (g/cm^3) * 1000 (kg/m^3 / g/cm^3) / m_nuc_MKS (kg) * 1e-45 (m^3 / fm^3)
n_central = rho_central_cgs * 1e3 / m_nuc_MKS * 1e-45

# Get dimensionless central energy density from EoS
epsilon_central_dim = Energy_from_n(n_central)
factor = 1/epsilon_central_dim

# Cantidades prima
rho_P_prima = lambda P: factor * rho_P_func(P/factor)
P_central_prima = factor * P_from_rho(1/factor)
epsilon_limit_prima = factor * epsilon_limit_dim

# Calculate physical central energy density for reference (MKS)
epsilon_central_MKS = epsilon_central_dim * rho_0_lambda_MKS

print(f"Central Density: {rho_central_cgs:.2e} g/cm^3")
print(f"Central Baryon Density: {n_central:.4e} fm^-3")
print(f"Dimensionless Central Energy: {epsilon_central_dim:.4f}")

# Integrate TOV
rf = 40 # Dimensionless radius (approx 50 * R_scale)
dr = 1e-3
print("Solving TOV...")
sol_fisico, sol_completa_dim, r_dim = tov.integrador(rf, dr, epsilon_central_MKS, rho_P_prima, P_central_prima, densidad_limite=epsilon_limit_prima, sol_completa=True)

# Extract physical arrays
# Scaling factors
P0_scale = epsilon_central_MKS # Pa
R_scale = np.sqrt(c_MKS**4 / (4*pi*G_MKS*epsilon_central_MKS)) # m
M_scale = 4*pi*R_scale**3 * epsilon_central_MKS / c_MKS**2 # kg

P_dim = sol_completa_dim[0]
m_dim = sol_completa_dim[1]
phi_dim = sol_completa_dim[2]
rho_dim = sol_completa_dim[3]

r_phys = r_dim * R_scale # m
m_phys = m_dim * M_scale # kg
P_phys = P_dim * P0_scale # Pa
rho_phys = rho_dim * epsilon_central_MKS # J/m^3 (Energy density)

# ==========================================
# Check Conditions
# ==========================================

# Convert to Geometrized Units (G=c=1) for conditions
# Length unit L = m
# Mass M_geom = G*M / c^2 (meters)
# Pressure P_geom = G*P / c^4 (m^-2)
# Density rho_geom = G*rho / c^4 (m^-2)

m_geom = m_phys * G_MKS / c_MKS**2
P_geom = P_phys * G_MKS / c_MKS**4
rho_geom = rho_phys * G_MKS / c_MKS**4
r_geom = r_phys

# 1) 2m/r < 1
compactness = 2 * m_geom / r_geom
# Avoid division by zero at r=0
compactness[0] = 0 

# 2) rho'(r) and P'(r) < 0
# Calculate derivatives with respect to r_phys (or r_geom, same thing)
drho_dr = np.gradient(rho_phys, r_phys)
dP_dr = np.gradient(P_phys, r_phys)

# 3) rho''(r) <= 0
d2rho_dr2 = np.gradient(drho_dr, r_phys)

# 4) R_tilde
# Formula: \tilde{R} = 2(m+4pi P r^3)/(r^2-2m r) + 4pi r^3/3 (\rho + P)(1+8pi P r^2)/(r - 2m)^2
# Note: r^2 - 2mr = r(r-2m)
# We use geometrized units
term1_num = 2 * (m_geom + 4*pi * P_geom * r_geom**3)
term1_den = r_geom * (r_geom - 2*m_geom)
# Avoid division by zero at r=0
with np.errstate(divide='ignore', invalid='ignore'):
    term1 = term1_num / term1_den
    
    term2_num = (4*pi * r_geom**3 / 3.0) * (rho_geom + P_geom) * (1 + 8*pi * P_geom * r_geom**2)
    term2_den = (r_geom - 2*m_geom)**2
    term2 = term2_num / term2_den
    
    R_tilde = term1 + term2

# Handle singularity at r=0
# As r->0, R_tilde -> 0 (linear in r)
R_tilde[0] = 0

# ==========================================
# Print Summary
# ==========================================
print("\n" + "="*40)
print("Condiciones de Aceptabilidad Física")
print("="*40)

# 1) 2m/r < 1
max_compactness = np.max(compactness)
print(f"1) 2m/r < 1: {'CUMPLE' if max_compactness < 1 else 'FALLA'}")
print(f"   Max 2m/r = {max_compactness:.4f}")

# 2) rho' < 0 and P' < 0
# Ignore the center point where derivative might be 0
rho_decreasing = np.all(drho_dr[1:] < 0)
P_decreasing = np.all(dP_dr[1:] < 0)
print(f"2) rho'(r) < 0: {'CUMPLE' if rho_decreasing else 'FALLA'}")
print(f"   P'(r) < 0:   {'CUMPLE' if P_decreasing else 'FALLA'}")

# 3) rho'' <= 0
# This is a strong condition. Often rho'' > 0 near surface?
# Let's check percentage
concave_down = d2rho_dr2 <= 0
percent_concave = np.sum(concave_down) / len(concave_down) * 100
radius_not_concave = r_phys[~concave_down][0] if np.any(~concave_down) else None
print(f"3) rho''(r) <= 0: {percent_concave:.1f}% de la estrella")
if radius_not_concave is not None:
    print(f"   Primera violación en r = {radius_not_concave/1000:.2f} km")

# 4) R_tilde > 0 and no sign change
min_R_tilde = np.min(R_tilde[1:]) # Skip center
sign_changes = np.sum(np.diff(np.sign(R_tilde[1:])) != 0)
print(f"4) R_tilde > 0: {'CUMPLE' if min_R_tilde > 0 else 'FALLA'}")
print(f"   Min R_tilde = {min_R_tilde:.4e} m^-1")
print(f"   Cambios de signo: {sign_changes}")

# ==========================================
# Plotting
# ==========================================
# Style from plot_descomposicion_energia.py
colors = ["k", "#914098", "#852B17", "#3E7C17"]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# fig.suptitle(r'Condiciones de Aceptabilidad Física ($\rho_c = 10^{15}$ g/cm$^3$)', fontsize=16)

# Plot 1: Compactness 2m/r
ax1 = axs[0, 0]
ax1.plot(r_phys/1000, compactness, color=colors[0], linewidth=2)
# ax1.axhline(1.0, color='k', linestyle='--', label='Límite Buchdahl (1.0)')
# ax1.set_xlabel(r'Radio (km)', fontsize=16)
ax1.set_ylabel(r'$2m/r$', fontsize=16)
# ax1.set_title(r'Compacidad ($2m/r < 1$)', fontsize=14)
ax1.grid(True, alpha=0.3)
# ax1.legend()

# Plot 2: Derivatives rho' and P'
ax2 = axs[0, 1]
# drho_dr_cgs = drho_dr * 0.1 # erg/cm^4
# dP_dr_cgs = dP_dr * 0.1 # Ba/cm
r_km = r_phys / 1000

ax2.plot(r_km, -drho_dr, color=colors[0], linestyle='-', label=r"$\rho'(r)$")
ax2.plot(r_km, -dP_dr, color=colors[2], linestyle='--', label=r"$P'(r)$")
# ax2.axhline(0, color='k', linestyle=':', linewidth=1)
# ax2.set_xlabel(r'Radio (km)', fontsize=16)
ax2.set_ylabel(r'(-)Derivadas (J/m$^4$)', fontsize=16)
# ax2.set_title(r'Gradientes de Presión y Densidad', fontsize=14)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=14)

# Plot 3: Second derivative rho''
ax3 = axs[1, 0]
# d2rho_dr2_cgs = d2rho_dr2 * 0.001 # erg/cm^5

ax3.plot(r_km, -d2rho_dr2, color=colors[0], linewidth=2)
# ax3.axhline(0, color='k', linestyle=':', linewidth=1)
ax3.set_xlabel(r'Radio (km)', fontsize=16)
ax3.set_ylabel(r"(-)$\rho''(r)$ (J/m$^5$)", fontsize=16)
# ax3.set_title(r'Concavidad de la Densidad ($\rho" \leq 0$)', fontsize=14)
# ax3.set_ylim(-1e29,1e29)
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# Plot 4: R_tilde
ax4 = axs[1, 1]
# Convert R_tilde to km^-1 for better readability
R_tilde_km = R_tilde * 1000
ax4.plot(r_km[1:], R_tilde_km[1:], color=colors[0], linewidth=2) # Skip r=0
# ax4.axhline(0, color='k', linestyle=':', linewidth=1)
ax4.set_xlabel(r'Radio (km)', fontsize=16)
ax4.set_ylabel(r'$\tilde{R}$ (km$^{-1}$)', fontsize=16)
# ax4.set_title(r'Condición $\tilde{R} > 0$', fontsize=14)
ax4.grid(True, alpha=0.3)

for ax in axs.flat:
    ax.set_xlim(r_km[0], r_km[-1])

plt.tight_layout()
plt.show()
