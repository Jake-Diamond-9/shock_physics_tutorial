# Written by Jake Diamond with assistance from LLMs
#
# Johns Hopkins University
# Army Research Laboratory
# 
# November, 2026
#
# ORCiD:    0000-0001-7905-4260
# Email:    jdiamo15@jhu.edu
# Repo DOI: TBD
# 
# Associated paper: "Shock Physics and the Plate Impact Experiment: A Tutorial" 
# Paper DOI:        TBD

import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar


def shock_physics_tutorial(
    Mode="Impact",
    Verbose=True,
    Velocity=0.0,
    Pressure=0.0,
    VolumeRatio=0.0,
    Duration=0.0,
    ProjName="",
    ProjThick=0.0,
    ProjTemp=298.0,
    TargName="Cu",
    TargThick=5e-3,
    TargTemp=298.0
):
    """
    SHOCK_PHYSICS_TUTORIAL Calculates shock states using keyword arguments.

    Usage Examples:
    -------------------------------------------------------------------------
    IMPACT MODE:
      proj, targ = shock_physics_tutorial(Mode="Impact", Velocity=3000, ProjName="Ta", ProjThick=1e-3, TargName="Cu", TargThick=5e-3, Verbose=True)

    PRESSURE MODE:
      targ = shock_physics_tutorial(Mode="Pressure", Pressure=100e9, Duration=0.5e-6, TargName="Al 2024", TargThick=5e-3, Verbose=True)

    VOLUME MODE:
      targ = shock_physics_tutorial(Mode="Volume", VolumeRatio=0.85, Duration=0.5e-6, TargName="Mo", TargThick=5e-3, Verbose=True)
    -------------------------------------------------------------------------
    """
    valid_modes = ["impact", "pressure", "volume"]
    mode_lower = Mode.lower()
    if mode_lower not in valid_modes:
        raise ValueError(f"Mode must be one of {['Impact', 'Pressure', 'Volume']}.")

    # 1. Unit Conversion Setup ----------------------------------------------
    # [Velocity (m/s -> cm/us), Density (kg/m^3 -> g/cm^3), Energy (J/kg -> Mbar*cm^3/g), 1, Length (m -> cm), Time (s -> us)]
    uc = [1e4, 1e3, 1e8, 1.0, 1e-2, 1e-6]

    # 2. Validation & Geometry Loading --------------------------------------
    if mode_lower == "impact":
        if Velocity == 0 or ProjName == "" or ProjThick == 0:
            raise ValueError("Impact Mode requires: Velocity, ProjName, and ProjThick.")
        
        calc_projectile = True
        mat_P = load_mat(ProjName, uc, ProjTemp)
        h_P = ProjThick / uc[4]
        x_dot_P = Velocity / uc[0]

        if Verbose:
            print("======================================================")
            print("SHOCK PHYSICS TOOL: IMPACT MODE")
            print("======================================================")
            print(f"Impact Velocity: {Velocity:.2f} m/s")
            print(f"Projectile: {ProjName} ({ProjThick*1e3:.2f} mm) | Target: {TargName} ({TargThick*1e3:.2f} mm)\n")

    elif mode_lower == "pressure":
        if Pressure == 0 or Duration == 0:
            raise ValueError("Pressure Mode requires: Pressure (Pa) and Duration (s).")
        
        calc_projectile = False
        driver_val = Pressure / 1e11  # Pa -> Mbar
        t_pulse_defined = Duration / uc[5]

        if Verbose:
            print("======================================================")
            print("SHOCK PHYSICS TOOL: PRESSURE MODE")
            print("======================================================")
            print(f"Target Pressure: {Pressure/1e9:.2f} GPa")
            print(f"Target: {TargName} ({TargThick*1e3:.2f} mm)\n")

    elif mode_lower == "volume":
        if VolumeRatio == 0 or Duration == 0:
            raise ValueError("Volume Mode requires: VolumeRatio (v/v0) and Duration (s).")
        if VolumeRatio >= 1.0:
            raise ValueError("Volume Ratio must be < 1.0 for compression.")
        
        calc_projectile = False
        driver_val = VolumeRatio
        t_pulse_defined = Duration / uc[5]

        if Verbose:
            print("======================================================")
            print("SHOCK PHYSICS TOOL: VOLUME MODE")
            print("======================================================")
            print(f"Volume Ratio:    {VolumeRatio:.4f}")
            print(f"Target: {TargName} ({TargThick*1e3:.2f} mm)\n")

    # Load Target (Common to all modes)
    mat_T = load_mat(TargName, uc, TargTemp)
    h_T = TargThick / uc[4]

    # 3. The Calculation Branch ---------------------------------------------
    if calc_projectile:
        # === PATH A: IMPACT MODE (up -> P) ===
        rhoP, CP, SP = mat_P["rho0"], mat_P["C0"], mat_P["S"]
        rhoT, CT, ST = mat_T["rho0"], mat_T["C0"], mat_T["S"]

        if rhoP == rhoT and CP == CT and SP == ST:
            # Symmetric Impact
            x_dot = 0.5 * x_dot_P
        else:
            # Unsymmetric Impact
            term1 = 1.0 / (2.0 * (rhoP * SP - rhoT * ST))
            term2 = rhoP * CP + rhoT * CT + 2.0 * x_dot_P * rhoP * SP
            term3 = (rhoP * CP + rhoT * CT)**2 + 4.0 * x_dot_P * rhoP * rhoT * (CP * ST + CT * SP + SP * ST * x_dot_P)
            x_dot = term1 * (term2 - math.sqrt(term3))
            print("********************* WARNING *********************")
            print("Code assumes a releasae to zero pressure in both the projectile and target!")
            print("This is accurate behind the free surfaces, not necessarily at the impact interface!\n")


        # Target Shock Velocity
        U_ST = CT + ST * x_dot

        # Projectile Effective Velocities
        x_dot_P_effective = x_dot_P - x_dot
        U_SP_effective = CP + SP * x_dot_P_effective

        # Calculate Pressure
        p_T = rhoT * U_ST * x_dot
        p_P = rhoP * U_SP_effective * x_dot_P_effective

        # Calculate Specific Volumes
        v_T = mat_T["v0"] * (1.0 - x_dot / U_ST)
        v_P = mat_P["v0"] * (1.0 - x_dot_P_effective / U_SP_effective)

    elif mode_lower == "pressure":
        # === PATH B: PRESSURE MODE (P -> up) ===
        p_T = driver_val
        rhoT, CT, ST, v0T = mat_T["rho0"], mat_T["C0"], mat_T["S"], mat_T["v0"]

        term_sqrt = math.sqrt((rhoT * CT**2 + 4.0 * ST * p_T) / rhoT)
        num = 2.0 * ST * p_T + CT**2 * rhoT - CT * rhoT * term_sqrt
        den = 2.0 * ST**2 * p_T * rhoT
        v_T = v0T - num / den

        x_dot = (CT * (v0T - v_T)) / (v0T - ST * (v0T - v_T))
        U_ST = CT + ST * x_dot

    elif mode_lower == "volume":
        # === PATH C: VOLUME MODE (J -> up -> P) ===
        J_T = driver_val
        v_T = mat_T["v0"] * J_T

        Delta_T = 1.0 - J_T
        x_dot = (mat_T["C0"] * Delta_T) / (1.0 - mat_T["S"] * Delta_T)
        U_ST = mat_T["C0"] + mat_T["S"] * x_dot
        p_T = mat_T["rho0"] * U_ST * x_dot

    # 4. Shared Physics Calculations (Thermodynamics & Release) -------------
    # --- TARGET CALCULATIONS ---
    J_T = v_T / mat_T["v0"]
    eps_jump_T = 0.5 * p_T * (mat_T["v0"] - v_T)
    theta_T, eta_jump_T = hugoniot_temp_ent(v_T, mat_T)
    v_2T, eps_jump_2T, C_L_head_T, C_L_tail_T, x_dot_fs_T = isentrope_release(v_T, p_T, x_dot, mat_T)
    J_2T = v_2T / mat_T["v0"]
    theta_2T = isentrope_temp(mat_T, theta_T, v_T, v_2T)

    # --- PROJECTILE CALCULATIONS (If Impact Mode) ---
    if calc_projectile:
        J_P = v_P / mat_P["v0"]
        eps_jump_P = 0.5 * p_P * (mat_P["v0"] - v_P)
        theta_P, eta_jump_P = hugoniot_temp_ent(v_P, mat_P)
        v_2P, eps_jump_2P, C_L_head_P, C_L_tail_P, _ = isentrope_release(v_P, p_P, x_dot, mat_P)
        J_2P = v_2P / mat_P["v0"]
        theta_2P = isentrope_temp(mat_P, theta_P, v_P, v_2P)

        t_pulse_shock = h_P / U_SP_effective + h_P / C_L_head_P
    else:
        t_pulse_shock = t_pulse_defined

    # --- TIMING CALCULATIONS (Target) ---
    h_o = t_pulse_shock * ((1.0 / U_ST - 1.0 / C_L_head_T)**(-1))
    t_pulse_fs = (h_o - h_T) * np.sqrt(C_L_head_T/C_L_tail_T) * (1/U_ST - 1/C_L_head_T)
    if t_pulse_fs < 0:
        t_pulse_fs = 0.0

    # 5. Display Output & Return Dictionaries -------------------------------
    # Unit Multipliers for Printing (Readable Units)
    u_vel_p, u_press_p, u_eng_p, u_len_p, u_time_p = 1e4, 100.0, 1e5, 10.0, 1000.0
    lbl_vel_p, lbl_press_p, lbl_eng_p, lbl_len_p, lbl_time_p = "m/s", "GPa", "J/g", "mm", "ns"

    # Unit Multipliers for Dictionary (SI)
    u_vel_d, u_press_d, u_eng_d, u_len_d, u_time_d = 1e4, 1e11, 1e8, 1e-2, 1e-6

    # --- PROJECTILE OUTPUT (Impact Mode Only) ---
    if calc_projectile:
        proj_dict = {
            "Effective Particle Velocity": x_dot_P_effective * u_vel_d,
            "Effective Shock Velocity": U_SP_effective * u_vel_d,
            "Pressure": p_P * u_press_d,
            "Volume Ratio": J_P,
            "Specific Internal Energy Jump": eps_jump_P * u_eng_d,
            "Temperature": theta_P,
            "Specific Entropy Jump": eta_jump_P * 1e8,
            "Release Volume Ratio": J_2P,
            "Release Specific Internal Energy Difference": eps_jump_2P * u_eng_d,
            "Release Temperature": theta_2P,
            "Release Head Speed": C_L_head_P * u_vel_d,
            "Release Tail Speed": C_L_tail_P * u_vel_d,
            "Shock Pulse Duration": t_pulse_shock * u_time_d
        }

        if Verbose:
            print(f"=== PROJECTILE ({mat_P['name']}) ===")
            print_res("Effective Particle Velocity", x_dot_P_effective, u_vel_p, lbl_vel_p)
            print_res("Effective Shock Velocity", U_SP_effective, u_vel_p, lbl_vel_p)
            print_res("Pressure", p_P, u_press_p, lbl_press_p)
            print_res("Volume Ratio", J_P, 1.0, "")
            print_res("Specific Internal Energy Jump", eps_jump_P, u_eng_p, lbl_eng_p)
            print_res("Temperature", theta_P, 1.0, "K")
            print_res("Specific Entropy Jump", eta_jump_P, 1e8, "J/(kg K)")
            print_res("Release Volume Ratio", J_2P, 1.0, "")
            print_res("Release Internal Energy Difference", eps_jump_2P, u_eng_p, lbl_eng_p)
            print_res("Release Temperature", theta_2P, 1.0, "K")
            print_res("Release Head Speed", C_L_head_P, u_vel_p, lbl_vel_p)
            print_res("Release Tail Speed", C_L_tail_P, u_vel_p, lbl_vel_p)
            print_res("Shock Pulse Duration", t_pulse_shock, u_time_p, lbl_time_p)
            print()

    # --- TARGET OUTPUT (All Modes) ---
    targ_dict = {
        "Particle Velocity": x_dot * u_vel_d,
        "Shock Velocity": U_ST * u_vel_d,
        "Pressure": p_T * u_press_d,
        "Volume Ratio": J_T,
        "Specific Internal Energy Jump": eps_jump_T * u_eng_d,
        "Temperature": theta_T,
        "Specific Entropy Jump": eta_jump_T * 1e8,
        "Release Volume Ratio": J_2T,
        "Release Specific Internal Energy Difference": eps_jump_2T * u_eng_d,
        "Release Temperature": theta_2T,
        "Release Head Speed": C_L_head_T * u_vel_d,
        "Release Tail Speed": C_L_tail_T * u_vel_d,
        "Free Surface Velocity": x_dot_fs_T * u_vel_d,
        "Shock Overtake Thickness": h_o * u_len_d,
        "Free Surface Pulse Duration": t_pulse_fs * u_time_d
    }

    if Verbose:
        print(f"=== TARGET ({mat_T['name']}) ===")
        print_res("Particle Velocity", x_dot, u_vel_p, lbl_vel_p)
        print_res("Shock Velocity", U_ST, u_vel_p, lbl_vel_p)
        print_res("Pressure", p_T, u_press_p, lbl_press_p)
        print_res("Volume Ratio", J_T, 1.0, "")
        print_res("Specific Internal Energy Jump", eps_jump_T, u_eng_p, lbl_eng_p)
        print_res("Temperature", theta_T, 1.0, "K")
        print_res("Specific Entropy Jump", eta_jump_T, 1e8, "J/(kg K)")
        print_res("Release Volume Ratio", J_2T, 1.0, "")
        print_res("Release Internal Energy Difference", eps_jump_2T, u_eng_p, lbl_eng_p)
        print_res("Release Temperature", theta_2T, 1.0, "K")
        print_res("Release Head Speed", C_L_head_T, u_vel_p, lbl_vel_p)
        print_res("Release Tail Speed", C_L_tail_T, u_vel_p, lbl_vel_p)
        print_res("Free Surface Velocity", x_dot_fs_T, u_vel_p, lbl_vel_p)
        print_res("Shock Overtake Thickness", h_o, u_len_p, lbl_len_p)
        print_res("Free Surface Pulse Duration", t_pulse_fs, u_time_p, lbl_time_p)

    if calc_projectile:
        return proj_dict, targ_dict
    else:
        return targ_dict


# HELPER FUNCTIONS -------------------------------------------------------

def print_res(label, val, mult, units):
    """Helper for pretty printing."""
    res = val * mult
    if units == "":
        print(f"{label:<40}: {res:.4f}")
    else:
        if abs(res) > 1e4 or (abs(res) < 1e-2 and res != 0):
            print(f"{label:<40}: {res:.4e} {units}")
        else:
            print(f"{label:<40}: {res:.4f} {units}")


def load_mat(name, uc, T_init):
    """Loads props from database and converts to sim units using a dictionary."""
    props = get_material_data(name)
    rho0 = props[0] / uc[1]
    return {
        "name": name,
        "rho0": rho0,
        "v0": 1.0 / rho0,
        "C0": props[1] / uc[0],
        "S": props[2],
        "gamma0": props[3],
        "Cv": props[4] / uc[2],
        "T0": T_init
    }


def hugoniot_temp_ent(v, mat):
    """Eq 38 & 41: Temperature and specific entropy jump along the Hugoniot."""
    v_R = mat["v0"]
    S = mat["S"]
    g0 = mat["gamma0"]
    C0 = mat["C0"]
    Cv = mat["Cv"]
    T0 = mat["T0"]
    rhoR = mat["rho0"]

    def chi_c(V):
        return np.exp((g0 / v_R) * (v_R - V))

    def theta_integrand(V):
        return (1.0 / chi_c(V)) * (((v_R - V)**2) / ((1.0 - rhoR * S * (v_R - V))**3))

    int_opts = {"epsrel": 1e-4, "epsabs": 1e-6}

    # Calculate Temperature
    int_val, _ = quad(theta_integrand, v_R, v, **int_opts)
    theta = chi_c(v) * (T0 - ((rhoR**3 * S * C0**2) / Cv) * int_val)

    # Calculate specific entropy jump (Eq 41)
    def theta_fun(V):
        val, _ = quad(theta_integrand, v_R, V, **int_opts)
        return chi_c(V) * (T0 - ((rhoR**3 * S * C0**2) / Cv) * val)

    def eta_integrand(V):
        return (1.0 / theta_fun(V)) * (((v_R - V)**2) / ((1.0 - rhoR * S * (v_R - V))**3))

    eta_int_val, _ = quad(eta_integrand, v_R, v, **int_opts)
    eta_jump = -(rhoR**3 * S * C0**2) * eta_int_val

    return theta, eta_jump


def isentrope_release(v, p, x_dot, mat):
    """Calculates state variables along the isentropic release path to P=0."""
    v_R = mat["v0"]
    S = mat["S"]
    g0 = mat["gamma0"]
    C0 = mat["C0"]
    rhoR = mat["rho0"]

    int_opts = {"epsrel": 1e-4, "epsabs": 1e-6}

    def p_H(V):
        return (rhoR * C0**2 * ((v_R - V) / v_R)) / (1.0 - S * ((v_R - V) / v_R))**2

    def dpHdv(V):
        return -(C0**2 * (v_R - S * V + S * v_R)) / (v_R + S * V - S * v_R)**3

    def kappa_c(V):
        return 0.5 * (g0 / v_R) * p_H(V) + (1.0 - 0.5 * (g0 / v_R) * (v_R - V)) * dpHdv(V)

    def chi_c(V):
        return np.exp((g0 / v_R) * (v - V))

    def p_eta_integrand(V):
        return kappa_c(V) / chi_c(V)

    def p_eta_func(V):
        val, _ = quad(p_eta_integrand, v, V, **int_opts)
        return chi_c(V) * (p_H(v) + val)

    # Find Released Volume v_2 where P_is(v_2) = 0 (Eq 53)
    sol = root_scalar(p_eta_func, x0=v_R, x1=v_R * 1.05, method="secant")
    v_2 = sol.root

    # Calculate specific internal energy difference in the released state (Eq 54)
    eps_release_integral, _ = quad(p_eta_func, v, v_2, **int_opts)
    eps_shock = 0.5 * p * (v_R - v)
    eps_jump_2 = eps_shock - eps_release_integral

    # Lagrangian Sound Speed (C_L, Eq 61)
    def C_L_fun(V):
        return math.sqrt((1.0 / rhoR**2) * ((g0 / v_R) * p_eta_func(V) - kappa_c(V)))

    C_L_head = C_L_fun(v)
    C_L_tail = C_L_fun(v_2)

    # Free Surface Velocity (Eq 69)
    fs_int, _ = quad(C_L_fun, v, v_2, **int_opts)
    x_dot_fs = x_dot + (1.0 / v_R) * fs_int

    return v_2, eps_jump_2, C_L_head, C_L_tail, x_dot_fs


def isentrope_temp(mat, theta, v, v_2):
    """Temperature along the isentrope (Eq 58)."""
    return theta * math.exp((mat["gamma0"] / mat["v0"]) * (v - v_2))


def get_material_data(name):
    """
    Function to retrieve material properties (Stored in SI Units).
    Properties from Kinslow, 1970, http://dx.doi.org/10.1016/b978-0-124-08950-1.x5001-0
    format: [rho_0 (kg/m^3), C_0 (m/s), S, gamma_0, C_v (J/kg K)]
    """
    materials = {
        "Al 2024":      [2785.0, 5328.0, 1.338, 2.00,  838.0],
        "Be":           [1851.0, 7998.0, 1.124, 1.16,  1755.0],
        "Cr":           [7117.0, 5173.0, 1.473, 1.19,  444.4],
        "Cu":           [8930.0, 3940.0, 1.489, 1.99,  371.8],
        "Hf":           [12885.0, 2954.0, 1.121, 0.98, 149.3],
        "Ir":           [22484.0, 3916.0, 1.457, 1.97, 132.8],
        "Mg":           [1740.0, 4492.0, 1.263, 1.425, 993.3],
        "Mg AZ31B":     [1780.0, 4522.0, 1.242, 1.434, 963.8],
        "Mo":           [10206.0, 5124.0, 1.233, 1.52, 245.7],
        "Nb":           [8586.0, 4438.0, 1.207, 1.470, 265.3],
        "Pd":           [11991.0, 3948.0, 1.588, 2.26, 233.4],
        "Pt":           [21419.0, 3598.0, 1.544, 2.40, 127.0],
        "Re":           [21020.0, 4184.0, 1.367, 2.44, 135.6],
        "Rh":           [12428.0, 4807.0, 1.376, 1.88, 238.8],
        "SS 304":       [7896.0, 4569.0, 1.490, 2.17,  441.3],
        "Ta":           [16654.0, 3414.0, 1.200, 1.60, 137.4],
        "Ti":           [4528.0, 5220.0, 0.767, 1.09,  518.6],
        "V":            [6100.0, 5077.0, 1.201, 1.29,  479.1],
        "W":            [19224.0, 4029.0, 1.237, 1.54, 131.0],
        "Zr":           [6505.0, 3757.0, 1.018, 1.09,  272.1],
        "SiC":          [3120.0, 8000.0, 0.950, 1.25,  646.3],
        "WC":           [15020.0, 4920.0, 1.339, 1.50, 207.7],
        "Epoxy 828":    [1198.0, 2678.0, 1.520, 1.13,  1042.0],
        "LDPE":         [915.0,  2901.0, 1.481, 1.644, 2300.0],
        "Neoprene":     [1439.0, 2785.0, 1.419, 1.39,  1649.0],
        "Paraffin":     [900.0,  2960.0, 2.531, 1.18,  2890.0],
        "PMMA":         [1185.0, 2572.0, 1.536, 0.97,  1469.0],
        "Polyurethane": [1265.0, 2486.0, 1.577, 1.55,  1806.0],
        "Polyrubber":   [1010.0, 852.0,  1.865, 1.50,  1990.0],
        "PS":           [1044.0, 2746.0, 1.319, 1.18,  1345.0],
        "PTFE":         [2147.0, 1682.0, 1.819, 0.59,  1048.0],
        "PVC":          [1681.0, 1952.0, 1.660, 0.87,  2090.0],
    }
    
    if name not in materials:
        raise ValueError(f"Material '{name}' not found in database.")
    return materials[name]