import numpy as np
from scipy.interpolate import PchipInterpolator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import matplotlib.pyplot as plt
import os
import sys

# Save and restore terminal working directory
_prev_cwd = os.getcwd()
# Ensure the working directory is this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)
import IsospinEoS as isoEoS
import NSMatterEoS as nsEoS
import ResolverTOV as tov
# Restore the original working directory
os.chdir(_prev_cwd)

# Physical constants in MKS and conversion factors
hbar_MKS = 1.0545718e-34  # J s
c_MKS = 299792458        # m/s
G_MKS = 6.67430e-11      # m^3/kg/s^2
proton_mass = 1.6726219e-27  # kg
neutron_mass = 1.6749275e-27 # kg
m_nuc_MKS = (proton_mass + neutron_mass) / 2.0  # kg
e_MKS = 1.602186021766e-19  # C
# Conversion factors
Kg_to_fm11 = c_MKS / hbar_MKS * 1e-15
MeV_to_fm11 = e_MKS / (hbar_MKS * c_MKS * 1e9)
MeVfm_to_Jm = 1e51 * e_MKS
# Derived natural units
m_nuc = m_nuc_MKS * Kg_to_fm11

# Top-level workers for multiprocessing (pickleable)
def _nuclear_worker(args):
	i, j, A_sigma, A_omega, params, n_prove = args
	p = params.copy()
	p[0], p[1] = A_sigma, A_omega
	p.append(0.0) 
	sat, ebind, Kc, asym, Lc = isoEoS.calculate_properties(n_prove, p)
	return i, j, sat, ebind, Kc, asym, Lc

def _stellar_worker(args):
    i, j, A_sigma, A_omega, params, target_mass = args
    p = params.copy()
    p[0], p[1] = A_sigma, A_omega
    # Build EoS
    dens_max = 1e18 * 1e3 * (1e-45 / m_nuc_MKS)
    dens_min = 1e12 * 1e3 * (1e-45 / m_nuc_MKS)
    n_range = np.logspace(np.log10(dens_min), np.log10(dens_max), 200)
    crust_file = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'EoS_tables', 'EoS_crust.txt'))
    rho_P, pres, ener, dens_sirve, *_ = nsEoS.EoS(n_range, p, add_crust=True, crust_file_path=crust_file)
    dens_lim = ener[0]
    P_rho = PchipInterpolator(ener, pres)
    energia_densidad = PchipInterpolator(dens_sirve, ener)
    rhos_central = np.logspace(13.5, 15.5, 150)
    masses = np.zeros_like(rhos_central)
    radios = np.zeros_like(rhos_central)
    compacs = np.zeros_like(rhos_central)
    for k, rho_m in enumerate(rhos_central):
        n_bar = rho_m * 1e3 / m_nuc_MKS * 1e-45
        rho0_dim = energia_densidad(n_bar)
        R = 1.0 / rho0_dim
        rho_P_pr = lambda P: R * rho_P(P / R)
        P_central_pr = R * P_rho(1 / R)
        dens_lim_pr = R * dens_lim
        rho_nat_to_MKS = 1.0 / MeV_to_fm11 * MeVfm_to_Jm
        sol = tov.integrador(rf=30.0, dr=1e-3,
                            rho0=rho0_dim * m_nuc**4 / 2 * rho_nat_to_MKS,
                            rho_P=rho_P_pr, P_central=P_central_pr,
                            densidad_limite=dens_lim_pr)
        r_phys, m_phys, *_ = sol
        compacs[k] = G_MKS * m_phys / c_MKS**2 / r_phys
        radios[k] = r_phys * 1e-3
        masses[k] = m_phys / 1.989e30
    idx_max = np.nanargmax(masses)
    mass_max = masses[idx_max]
    comp_max = compacs[idx_max]
    # interpolate radius for target_mass
    if masses[0] <= target_mass <= masses[-1]:
        idx = np.searchsorted(masses, target_mass)
        m1, m2 = masses[idx-1], masses[idx]
        r1, r2 = radios[idx-1], radios[idx]
        radius_can = r1 + (target_mass - m1) / (m2 - m1) * (r2 - r1)
    else:
        radius_can = np.nan
    return i, j, mass_max, comp_max, radius_can

def compute_nuclear_mesh(A_sigma_range, A_omega_range, params,
						 mask=None, n_prove=None, bounds=None):
	"""
	Compute mesh of saturation density (n_sat), binding energy (B/A), compression modulus (K),
	symmetry energy (a_sym) and slope L over ranges of A_sigma and A_omega.
	Returns 6 arrays: n_sat_mesh, ebind_mesh, K_mesh, a_sym_mesh, L_mesh and mask.
	If mask is None, computes full grid and then applies ranges to generate mask.
	
	Parameters
	----------
	bounds : dict, optional
		Dictionary with optional keys 'n_sat', 'ebind', 'K', 'a_sym', 'L', each containing
        a list [min, max]. Points outside these ranges will be excluded from the mesh and mask.
	"""
	if n_prove is None:
		n_prove = np.linspace(1e-3, 0.25, 200)
	
	# Extract bounds if provided
	K_bounds = bounds.get('K') if bounds is not None else None
	a_sym_bounds = bounds.get('a_sym') if bounds is not None else None
	L_bounds = bounds.get('L') if bounds is not None else None
	sat_min, sat_max = bounds.get('n_sat', (0.15, 0.18)) if bounds is not None else (0.15, 0.18)
	ebind_min, ebind_max = bounds.get('ebind', (-18.0, -12.0)) if bounds is not None else (-18.0, -12.0)

	A_sigma_mesh, A_omega_mesh = np.meshgrid(A_sigma_range, A_omega_range)
	shape = A_sigma_mesh.shape
	n_sat_mesh = np.full(shape, np.nan)
	ebind_mesh = np.full(shape, np.nan)
	K_mesh = np.full(shape, np.nan)
	a_sym_mesh = np.full(shape, np.nan)
	L_mesh = np.full(shape, np.nan)
	# Prepare tasks (i, j, A_sigma, A_omega, params, n_prove)
	coords = list(zip(*np.where(mask))) if mask is not None else [(i, j) for i in range(shape[0]) for j in range(shape[1])]
	tasks = [(i, j, A_sigma_mesh[i, j], A_omega_mesh[i, j], params, n_prove) for i, j in coords]
	# Parallelize using ProcessPoolExecutor and top-level worker
	max_workers = os.cpu_count() or 8
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		for i, j, sat, ebind, Kc, asym, Lc in executor.map(_nuclear_worker, tasks):
			n_sat_mesh[i, j] = sat
			ebind_mesh[i, j] = ebind
			
			# Check if sat and ebind are within basic ranges
			if sat_min <= sat <= sat_max and ebind_min <= ebind <= ebind_max:
				# Apply additional bounds if specified
				valid = True
				
				if K_bounds is not None:
					if not (K_bounds[0] <= Kc <= K_bounds[1]):
						valid = False
				
				if a_sym_bounds is not None:
					if not (a_sym_bounds[0] <= asym <= a_sym_bounds[1]):
						valid = False
				
				if L_bounds is not None:
					if not (L_bounds[0] <= Lc <= L_bounds[1]):
						valid = False
				
				# Only assign values if all bounds are satisfied
				if valid:
					K_mesh[i, j] = Kc
					a_sym_mesh[i, j] = asym
					L_mesh[i, j] = Lc
	
	if mask is None:
		# Build mask from finite K, a_sym, L values (which already satisfy all constraints)
		mask = (n_sat_mesh >= sat_min) & (n_sat_mesh <= sat_max) & \
			   (ebind_mesh >= ebind_min) & (ebind_mesh <= ebind_max) & \
			   np.isfinite(K_mesh) & np.isfinite(a_sym_mesh) & np.isfinite(L_mesh)
	return n_sat_mesh, ebind_mesh, K_mesh, a_sym_mesh, L_mesh, mask

def compute_stellar_mesh(A_sigma_range, A_omega_range, params, mask, target_mass=1.4):
    """
    Compute mesh of maximum mass, compactness and canonical radius (for target_mass in M_sun)
    over ranges of A_sigma and A_omega according to a boolean mask.
    Returns mass_mesh, comp_mesh, radius_mesh arrays.
    """
    # Create grids
    A_sigma_mesh, A_omega_mesh = np.meshgrid(A_sigma_range, A_omega_range)
    shape = A_sigma_mesh.shape
    mass_mesh = np.full(shape, np.nan)
    comp_mesh = np.full(shape, np.nan)
    radius_mesh = np.full(shape, np.nan)
    # Prepare tasks
    coords = list(zip(*np.where(mask)))
    tasks = [(i, j, A_sigma_mesh[i, j], A_omega_mesh[i, j], params, target_mass)
             for i, j in coords]
    # Parallel compute using ProcessPoolExecutor
    max_workers = os.cpu_count() or 8
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, j, mmax, comp, rcan in executor.map(_stellar_worker, tasks):
            mass_mesh[i, j] = mmax
            comp_mesh[i, j] = comp
            radius_mesh[i, j] = rcan
    return mass_mesh, comp_mesh, radius_mesh

def plot_nuclear_mesh(A_sigma_range, A_omega_range, n_sat_mesh, ebind_mesh, K_mesh, a_sym_mesh,
                      L_mesh, params, bounds=None, manual_label_position=5, equal_aspect=False, 
                      xlim=None, ylim=None, colormap=plt.cm.RdPu, contours=True):
    A_sigma_mesh, A_omega_mesh = np.meshgrid(A_sigma_range, A_omega_range)
    K_masked = np.ma.masked_invalid(K_mesh)
    sat_masked = np.ma.masked_invalid(n_sat_mesh)
    ebind_masked = np.ma.masked_invalid(ebind_mesh)
    a_sym_masked = np.ma.masked_invalid(a_sym_mesh)
    L_masked = np.ma.masked_invalid(L_mesh)

    sat_min, sat_max = bounds.get('n_sat', (0.15, 0.18)) if bounds is not None else (0.15, 0.18)
    ebind_min, ebind_max = bounds.get('ebind', (-18.0, -12.0)) if bounds is not None else (-18.0, -12.0)

    # Calculamos el area de la zona válida en el espacio A_sigma - A_omega
    # valid_area = np.sum(np.isfinite(K_masked)) * (A_sigma_range[1] - A_sigma_range[0]) * (A_omega_range[1] - A_omega_range[0])
    
    params_names = [r'$A_\rho$', r'$b$', r'$c$', 'Area']
    # params_text = '\n'.join([f'{name}={value:.3f}' if name not in [r'$b$', r'$c$'] else f'{name}={value:.3e}' for name, value in zip(params_names, params[2:]+[valid_area])])
    params_text = '\n'.join([f'{name}={value:.3f}' if name not in [r'$b$', r'$c$'] else f'{name}={value:.3e}' for name, value in zip(params_names[:-1], params[2:])])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    cmap = colormap

    # subplot 1: K_sat
    sc1 = ax1.scatter(
        A_sigma_mesh, A_omega_mesh,
        c=K_masked, cmap=cmap, marker='o'
    )
    cb = plt.colorbar(sc1, ax=ax1)
    cb.set_label(r'Módulo de compresión (MeV)', fontsize=14)
    # ax1.set_title(r'$K_{sat}$')
    ax1.set_ylabel(r'$A_\omega$', fontsize=14)

    # subplot 2: a_sym
    sc2 = ax2.scatter(
        A_sigma_mesh, A_omega_mesh,
        c=a_sym_masked, cmap=cmap,
        vmin=np.nanmin(a_sym_mesh), vmax=np.nanmax(a_sym_mesh)
    )
    cb = plt.colorbar(sc2, ax=ax2)
    cb.set_label('Coeficiente de simetría (MeV)', fontsize=14)
    # ax2.set_title(r'$a_{sym}$')
    
    # subplot 3: L
    sc3 = ax3.scatter(
        A_sigma_mesh, A_omega_mesh,
        c=L_masked, cmap=cmap,
        vmin=np.nanmin(L_mesh), vmax=np.nanmax(L_mesh)
    )
    cb = plt.colorbar(sc3, ax=ax3)
    cb.set_label('Pendiente de simetría (MeV)', fontsize=14)
    # ax3.set_title(r'$L_0$')
    
    for ax in (ax1, ax2, ax3):
        ax.text(0.05, 0.95, params_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(r'$A_\sigma$', fontsize=14)
        # ax.set_ylabel(r'$A_\omega$')
        if contours:
            cs1 = ax.contour(
                A_sigma_mesh, A_omega_mesh, sat_masked,
                levels=[sat_min, sat_max], colors='black', linestyles='-'
            )
            # Prepara posiciones manuales en los extremos de cada segmento
            manual_positions_sat = []
            for segs in cs1.allsegs:
                for seg in segs:
                    manual_positions_sat.append(tuple(seg[manual_label_position]))
            ax.clabel(
                cs1,
                fmt={sat_min: f'$n_0$={sat_min}', sat_max: f'$n_0$={sat_max}'},
                inline=True, fontsize=12, manual=manual_positions_sat
            )

            cs2 = ax.contour(
                A_sigma_mesh, A_omega_mesh, ebind_masked,
                levels=[ebind_min, ebind_max], colors='black', linestyles='-'
            )
            manual_positions_ebind = []
            for segs in cs2.allsegs:
                for seg in segs:
                    manual_positions_ebind.append(tuple(seg[manual_label_position]))
            ax.clabel(
                cs2,
                fmt={ebind_min: f'B/A={ebind_min}', ebind_max: f'B/A={ebind_max}'},
                inline=True, fontsize=12, manual=manual_positions_ebind
            )
        if equal_aspect:
            ax.set_aspect('equal', adjustable='box')
            
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.show()
    return fig

def plot_stellar_mesh(A_sigma_range, A_omega_range, mass_mesh, comp_mesh, radius_mesh, sat_mesh,
                      ebind_mesh, params, bounds=None, manual_label_position=5, equal_aspect=False, 
                      xlim=None, ylim=None, colormap=plt.cm.RdPu, contours=True):
    A_sigma_mesh, A_omega_mesh = np.meshgrid(A_sigma_range, A_omega_range)
    mass_masked = np.ma.masked_invalid(mass_mesh)
    comp_masked = np.ma.masked_invalid(comp_mesh)
    radius_masked = np.ma.masked_invalid(radius_mesh)
    sat_masked = np.ma.masked_invalid(sat_mesh)
    ebind_masked = np.ma.masked_invalid(ebind_mesh)
    
    # valid_area = np.sum(np.isfinite(mass_masked)) * (A_sigma_range[1] - A_sigma_range[0]) * (A_omega_range[1] - A_omega_range[0])
    
    params_names = [r'$A_\rho$', r'$b$', r'$c$', 'Area']
    # params_text = '\n'.join([f'{name}={value:.3f}' if name not in [r'$b$', r'$c$'] else f'{name}={value:.3e}' for name, value in zip(params_names, params[2:]+[valid_area])])
    params_text = '\n'.join([f'{name}={value:.3f}' if name not in [r'$b$', r'$c$'] else f'{name}={value:.3e}' for name, value in zip(params_names[:-1], params[2:])])

    sat_min, sat_max = bounds.get('n_sat', (0.15, 0.18)) if bounds is not None else (0.15, 0.18)
    ebind_min, ebind_max = bounds.get('ebind', (-18.0, -12.0)) if bounds is not None else (-18.0, -12.0)

    # Graficamos masa máxima y compacidad en dos paneles
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    cmap = colormap
    # Panel de masa máxima
    sc1 = ax1.scatter(A_sigma_mesh, A_omega_mesh, c=mass_masked, cmap=cmap, marker='o')
    cb = plt.colorbar(sc1, ax=ax1)
    cb.set_label(r'Masa máxima (M$_\odot$)', fontsize=14)
    # ax1.set_title('Masa máxima')
    ax1.set_ylabel(r'$A_\omega$', fontsize=14)

    # Panel de compacidad máxima
    sc2 = ax2.scatter(A_sigma_mesh, A_omega_mesh, c=comp_masked, cmap=cmap, marker='o')
    cb = plt.colorbar(sc2, ax=ax2)
    cb.set_label('Compacidad máxima ($GM/c^2R$)', fontsize=14)
    # ax2.set_title('Compacidad máxima')
        
    # Panel de radio canónico
    sc3 = ax3.scatter(A_sigma_mesh, A_omega_mesh, c=radius_masked, cmap=cmap, marker='o')
    cb = plt.colorbar(sc3, ax=ax3)
    cb.set_label('Radio canónico (km)', fontsize=14)
    # ax3.set_title(r'Radio canónico (A 1.4 M$_\odot$)')
    
    for ax in (ax1, ax2, ax3):
        ax.text(0.05, 0.95, params_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(r'$A_\sigma$', fontsize=14)
        # ax.set_ylabel(r'$A_\omega$')
        if contours:
            cs1 = ax.contour(
                A_sigma_mesh, A_omega_mesh, sat_masked,
                levels=[sat_min, sat_max], colors='black', linestyles='-'
            )
            # Prepara posiciones manuales en los extremos de cada segmento
            manual_positions_sat = []
            for segs in cs1.allsegs:
                for seg in segs:
                    manual_positions_sat.append(tuple(seg[manual_label_position]))
            ax.clabel(
                cs1,
                fmt={sat_min: f'$n_0$={sat_min}', sat_max: f'$n_0$={sat_max}'},
                inline=True, fontsize=12, manual=manual_positions_sat
            )

            cs2 = ax.contour(
                A_sigma_mesh, A_omega_mesh, ebind_masked,
                levels=[ebind_min, ebind_max], colors='black', linestyles='-'
            )
            manual_positions_ebind = []
            for segs in cs2.allsegs:
                for seg in segs:
                    manual_positions_ebind.append(tuple(seg[manual_label_position]))
            ax.clabel(
                cs2,
                fmt={ebind_min: f'B/A={ebind_min}', ebind_max: f'B/A={ebind_max}'},
                inline=True, fontsize=12, manual=manual_positions_ebind
            )
        if equal_aspect:
            ax.set_aspect('equal', adjustable='box')

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.show()
    return fig


# Quick sanity check
if __name__ == "__main__":
    import numpy as np
    from time import time

    # Test grid with pocos puntos
    A_sigma_test = np.linspace(190, 260, 10)
    A_omega_test = np.linspace(101, 130, 10)
 
	# Parameters
    params_test = [12.684*m_nuc**2, 7.148*m_nuc**2, (4.410*m_nuc**2)-12, 5.610e-3+0.0025, -6.986e-3]

    # Medir tiempo de propiedades nucleares
    t0 = time()
    n_sat, ebind, K, a_sym, L, mask = compute_nuclear_mesh(
        A_sigma_test, A_omega_test, params_test
    )
    t1 = time()
    print("Nuclear mesh shapes:",
          n_sat.shape, ebind.shape, K.shape, a_sym.shape, L.shape)
    # print("Mask:\n", mask)
    print(f"[Timing] compute_nuclear_mesh: {t1 - t0:.2f} s")

    # Medir tiempo de propiedades estelares
    t2 = time()
    mass, comp, rad = compute_stellar_mesh(
        A_sigma_test, A_omega_test, params_test, mask
    )
    t3 = time()
    print("Stellar mesh shapes:",
          mass.shape, comp.shape, rad.shape)
    print("Max mass:", np.max(mass[np.isfinite(mass)]), ", Masa min:", np.min(mass[np.isfinite(mass)]))
    print(f"[Timing] compute_stellar_mesh: {t3 - t2:.2f} s")
    # Graficar
    fig1 = plot_nuclear_mesh(
        A_sigma_test, A_omega_test,
        n_sat, ebind, K, a_sym, L,
        params_test
    )
    fig2 = plot_stellar_mesh(
        A_sigma_test, A_omega_test,
        mass, comp, rad,
        n_sat, ebind,
        params_test
    )