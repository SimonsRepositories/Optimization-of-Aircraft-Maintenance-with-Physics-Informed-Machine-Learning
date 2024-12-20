# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:09:04 2024

@author: aria
"""

import math
import h5py
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from dataclasses import dataclass, replace

# Define the EngineParameters class
@dataclass
class EngineParameters:
    # GAS PROPERTIES
    cp: float = 1.005
    cp_turbine: float = 1.15
    gamma: float = 1.4
    gamma_turbine: float = 1.333333333333

    # MAIN ENGINE PARAMETERS
    mass_flow: float = 100
    SOT: float = 1400
    bypass_ratio: float = 4.5
    fan_PR: float = 1.9
    LP_compressor_PR: float = 5.0
    HP_compressor_PR: float = 14.0/ 5

    # COMPONENT EFFICIENCIES
    fan_efficiency: float = 0.88
    LP_compressor_efficiency: float = 0.88
    HP_compressor_efficiency: float = 0.88
    LP_turbine_efficiency: float = 0.89
    HP_turbine_efficiency: float = 0.89
    combustion_efficiency: float = 0.999
    mechanical_efficiency: float = 0.995

    # FLIGHT CONDITIONS
    T_amb: float = 216.7
    P_amb: float = 22.628
    mach_number: float = 0.8

    # COMPONENT PRESSURE DROPS
    intake_pressure_loss: float = 0.005
    bypass_duct_pressure_loss: float = 0.03
    inter_compressor_pressure_loss: float = 0.02
    compressor_exit_diffuser_pressure_loss: float = 0.02
    combustor_pressure_loss: float = 0.03
    inter_turbine_duct_pressure_loss: float = 0.01
    jet_pipe_pressure_loss: float = 0.01

    # EXTRACTIONS AND COOLING FLOWS
    bleed_extraction: float = 0.01
    cooling_mechanical_loss: float = 0.005
    COMB_cooling_flow_percentage: float = 0.075
    TURB_cooling_flow_percentage: float = 0.08
    HPT_cooling_flow_percentage: float = 0.047
    LPT_cooling_flow_percentage: float = 0.033

# Define necessary functions
def compute_eta(PR, gamma, efficiency):
    numerator = PR ** ((gamma - 1) / gamma) - 1
    denominator = PR ** ((gamma - 1) / (gamma * efficiency)) - 1
    ETA = numerator / denominator if denominator != 0 else np.finfo(float).eps
    return ETA

def compute_eta_T_converge(T_in, T_out, ETAP, gamma, PR_initial=4, tol=1e-6, max_iter=100):
    PR = PR_initial
    iter_count = 0
    change = tol + 1  # Initialize change to a value larger than tolerance

    while change > tol and iter_count < max_iter:
        # Calculate ETA based on current PR
        ETA = (1 - PR ** (ETAP * (1 - gamma) / gamma)) / (1 - PR ** ((1 - gamma) / gamma))

        # Calculate adjusted PR based on temperatures and ETA
        PR_adjusted = (1 / (1 - (T_in - T_out) / (T_in * ETA))) ** (gamma / (gamma - 1))

        # Calculate the change and update PR
        change = abs(PR_adjusted - PR)
        PR = PR_adjusted
        iter_count += 1

    return ETA, PR

def compute_far(T31, T41, combustion_efficiency):
    # Fuel-Air Ratio Calculations
    FAR1 = 0.10118 + 2.00376E-05 * (700 - T31)
    FAR2 = 3.7078E-03 - 5.2368E-06 * (700 - T31) - 5.2632E-06 * T41
    FAR3 = 8.889E-08 * abs(T41 - 950)

    # Compute the discriminant for square root calculation
    discriminant = max(FAR1 ** 2 + FAR2, 0)
    sqrt_discriminant = math.sqrt(discriminant)

    # Calculate FAR with efficiency adjustment
    FAR = (FAR1 - sqrt_discriminant - FAR3) / combustion_efficiency
    return max(FAR, 0)  # Ensure FAR is not negative

def engine_simulation(params: EngineParameters, T_amb=None, P_amb=None, mach_number=None):
    # Use provided ambient conditions if given
    if T_amb is not None:
        params.T_amb = T_amb
    if P_amb is not None:
        params.P_amb = P_amb
    if mach_number is not None:
        params.mach_number = mach_number

    # Access each parameter explicitly
    # GAS PROPERTIES
    gamma = params.gamma
    gamma_turbine = params.gamma_turbine
    cp_turbine = params.cp_turbine
    cp = params.cp

    # FLIGHT CONDITIONS
    T_amb = params.T_amb
    P_amb = params.P_amb
    mach_number = params.mach_number

    # MAIN ENGINE PARAMETERS
    SOT = params.SOT
    mass_flow = params.mass_flow
    bypass_ratio = params.bypass_ratio
    fan_PR = params.fan_PR
    LP_compressor_PR = params.LP_compressor_PR
    HP_compressor_PR = params.HP_compressor_PR

    # COMPONENT EFFICIENCIES
    fan_efficiency = params.fan_efficiency
    LP_compressor_efficiency = params.LP_compressor_efficiency
    HP_compressor_efficiency = params.HP_compressor_efficiency
    combustion_efficiency = params.combustion_efficiency
    LP_turbine_efficiency = params.LP_turbine_efficiency
    HP_turbine_efficiency = params.HP_turbine_efficiency
    mechanical_efficiency = params.mechanical_efficiency

    # COMPONENT PRESSURE DROPS
    intake_pressure_loss = params.intake_pressure_loss
    bypass_duct_pressure_loss = params.bypass_duct_pressure_loss
    inter_compressor_pressure_loss = params.inter_compressor_pressure_loss
    compressor_exit_diffuser_pressure_loss = params.compressor_exit_diffuser_pressure_loss
    combustor_pressure_loss = params.combustor_pressure_loss
    inter_turbine_duct_pressure_loss = params.inter_turbine_duct_pressure_loss
    jet_pipe_pressure_loss = params.jet_pipe_pressure_loss

    # EXTRACTIONS AND COOLING FLOWS
    bleed_extraction = params.bleed_extraction
    cooling_mechanical_loss = params.cooling_mechanical_loss
    TURB_cooling_flow_percentage = params.TURB_cooling_flow_percentage
    COMB_cooling_flow_percentage = params.COMB_cooling_flow_percentage
    HPT_cooling_flow_percentage = params.HPT_cooling_flow_percentage
    LPT_cooling_flow_percentage = params.LPT_cooling_flow_percentage

    # FREE STREAM CONDITIONS
    T0 = (1 + ((gamma - 1) / 2) * mach_number ** 2) * T_amb  # K
    P0 = ((T0 / T_amb) ** (gamma / (gamma - 1))) * P_amb  # kPa
    W0 = mass_flow

    # Intake
    T2 = T0  # K
    P2 = P0 * (1 - intake_pressure_loss)  # kPa
    W2 = W0

    # FAN
    ETA2 = compute_eta(fan_PR, gamma, fan_efficiency)
    T24 = (T2 / ETA2) * (fan_PR ** ((gamma - 1) / gamma) - 1) + T2  # K
    T13 = T24  # K
    P13 = fan_PR * P2  # kPa
    P24 = P13  # kPa
    W24 = W0 / (1 + bypass_ratio)  # kg/s
    W13 = W0 - W24  # kg/s
    PW2 = W0 * cp * (T24 - T2)  # kW

    # BYPASS DUCT
    T17 = T13  # K
    P17 = (1 - bypass_duct_pressure_loss) * P13  # kPa
    W17 = W13  # kg/s

    # INTER COMPRESSOR DUCT
    T26 = T24  # K
    P26 = P24 * (1 - inter_compressor_pressure_loss)  # kPa
    W26 = W24  # kg/s

    # LP COMPRESSOR
    ETA_LPC = compute_eta(LP_compressor_PR, gamma, LP_compressor_efficiency)
    T27 = (T26 / ETA_LPC) * (LP_compressor_PR ** ((gamma - 1) / gamma) - 1) + T26  # K
    P27 = LP_compressor_PR * P26  # kPa
    W27 = W26
    PW_LPC = W26 * cp * (T27 - T24)  # kW
    Tbleed = T27
    cooling_flow_offtakes = W24 * cooling_mechanical_loss + bleed_extraction  # kg/s

    # HP COMPRESSOR
    ETA_HPC = compute_eta(HP_compressor_PR, gamma, HP_compressor_efficiency)
    T3 = (T27 / ETA_HPC) * (HP_compressor_PR ** ((gamma - 1) / gamma) - 1) + T27  # K
    P3 = HP_compressor_PR * P27  # kPa
    W3 = W27 - cooling_flow_offtakes # kg/s
    PW_HPC = W3 * cp * (T3 - T27)  # kW

    # COMPRESSOR EXIT DIFFUSER
    T31 = T3  # K
    P31 = P3 * (1 - compressor_exit_diffuser_pressure_loss)  # kPa
    W31 = W3 - W24 * TURB_cooling_flow_percentage - W24 * COMB_cooling_flow_percentage  # kg/s

    # COMBUSTOR AND SOT STATION
    T41 = SOT  # K
    P41 = P31 * (1 - combustor_pressure_loss)  # kPa
    FAR = compute_far(T31, T41, combustion_efficiency)  # Fuel-Air Ratio Calculations
    WF = FAR * (W31 + (mass_flow * COMB_cooling_flow_percentage))  # kg/s
    W41 = W31 + W24 * COMB_cooling_flow_percentage + WF  # kg/s
    W4 = W31 + WF  # kg/s
    T4 = (W41 * cp_turbine * T41 - W24 * COMB_cooling_flow_percentage * T3 * cp) / (W4 * cp_turbine)  # K

    # MIXING (inlet to HPT)
    P415 = P41  # kPa
    W415 = W41 + W24 * HPT_cooling_flow_percentage  # kg/s
    T415 = (W41 * cp_turbine * T41 + TURB_cooling_flow_percentage * cp * T31) / (W415 * cp_turbine)  # K

    # HP TURBINE
    PW415 = PW_HPC / mechanical_efficiency  # kW
    T416 = T415 - (PW415 / (W415 * cp_turbine))
    ETA415, P415Q416 = compute_eta_T_converge(T415, T416, HP_turbine_efficiency, gamma_turbine)
    ETA415 = max(ETA415, np.finfo(float).eps)
    P416 = P415 / P415Q416 if P415Q416 != 0 else np.finfo(float).eps  # kPa
    W416 = W415  # kg/s

    # No other cooling flows return, so (HP turbine exit, stage 44):
    P44 = P416  # kPa
    T44 = T416  # K
    W44 = W416  # kg/s

    # INTER TURBINE DUCT
    T46 = T44  # K
    P46 = P44 * (1 - inter_turbine_duct_pressure_loss)
    P46 = max(P46, np.finfo(float).eps)  # kPa
    W46 = W44  # kg/s

    # LP TURBINE
    PW46 = (PW2 + PW_LPC) / mechanical_efficiency  # kW
    T48 = T46 - (PW46 / (W46 * cp_turbine))
    ETA46, P46Q48 = compute_eta_T_converge(T46, T48, LP_turbine_efficiency, gamma_turbine)
    # (LP turbine exit, stage 48)
    P48 = P46 / P46Q48 if P46Q48 != 0 else np.finfo(float).eps  # kPa
    W48 = W46  # kg/s

    # COOLING AIR DOWNSTREAM OF TURBINE AND JET PIPE PRESSURE LOSS
    W5 = W46 + W24 * bleed_extraction + W24 * LPT_cooling_flow_percentage  # kg/s
    T5_numerator = (W48 * cp_turbine * T48 +
                    W26 * bleed_extraction * cp * Tbleed +
                    W26 * LPT_cooling_flow_percentage * cp * T31)
    T5_denominator = W5 * cp_turbine
    T5_denominator = max(T5_denominator, np.finfo(float).eps)
    T5 = T5_numerator / T5_denominator  # K
    #(core nozzle exit, stage 7)
    P7 = P48 * (1 - jet_pipe_pressure_loss)
    P7 = max(P7, np.finfo(float).eps)  # kPa

    # Collect results in a dictionary
    results = {
        "T_amb": T_amb, "P_amb": P_amb, "mach_number": mach_number,
        "T0": T0, "P0": P0, "W0": W0,
        "T2": T2, "P2": P2, "W2": W2,
        "ETA2": ETA2, "PW2": PW2,
        "T24": T24, "P24": P24, "W24": W24,
        "T13": T13, "P13": P13, "W13": W13,
        "T17": T17, "P17": P17, "W17": W17,
        "T26": T26, "P26": P26, "W26": W26,
        "T27": T27, "P27": P27, "W27": W27,
        "ETA_LPC": ETA_LPC, "PW_LPC": PW_LPC,
        "ETA_HPC": ETA_HPC, "PW_HPC": PW_HPC,
        "T3": T3, "P3": P3, "W3": W3,
        "Tbleed": Tbleed,
        "T31": T31, "P31": P31, "W31": W31,
        "T41": T41, "P41": P41,
        "FAR": FAR, "WF": WF,
        "W41": W41, "W4": W4, "T4": T4,
        "P415": P415, "W415": W415, "T415": T415, "PW415": PW415,
        "T416": T416, "ETA415": ETA415, "P415Q416": P415Q416, "P416": P416,
        "W416": W416, "P44": P44, "T44": T44, "W44": W44,
        "T46": T46, "P46": P46, "W46": W46,
        "PW46": PW46, "T48": T48, "ETA46": ETA46, "P46Q48": P46Q48, "P48": P48, "W48": W48,
        "W5": W5, "T5": T5, "P7": P7
    }

    # Round results for clarity
    for key in results:
        results[key] = round(results[key], 4) if isinstance(results[key], float) else results[key]

    return results

# Utility for Min-Max Normalization
def min_max_normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def min_max_denormalize(normalized, min_val, max_val):
    return normalized * (max_val - min_val) + min_val

# Renormalize bounds for normalized theta
def normalize_bounds(lower_bounds, upper_bounds, theta_min, theta_max):
    """
    Normalize bounds using Min-Max normalization.

    Args:
        lower_bounds: List of lower bounds for the parameters.
        upper_bounds: List of upper bounds for the parameters.
        theta_min: Minimum values for theta normalization.
        theta_max: Maximum values for theta normalization.

    Returns:
        List of normalized bounds for optimization.
    """
    norm_lower = (lower_bounds - theta_min) / (theta_max - theta_min)
    norm_upper = (upper_bounds - theta_min) / (theta_max - theta_min)
    return list(zip(norm_lower, norm_upper))

# Define the loss function
def loss_function(theta, engine_simulation, engine_params,
                  observed_data, parameter_names, keys, theta_min, theta_max):
    """
    Loss function with normalization for both theta and x_data.

    Args:
        theta: Parameters to optimize (normalized).
        engine_simulation: The simulation function.
        engine_params: Initial engine parameters dataclass.
        observed_data: Observed output data.
        parameter_names: List of parameter names to optimize.
        keys: Observable output variable keys.
        nominal_theta: Nominal values for theta parameters.

    Returns:
        Mean squared error (MSE) between normalized predicted and observed outputs.
    """
    # Denormalize theta
    theta_original = min_max_denormalize(theta, theta_min, theta_max)
    updated_params = replace(engine_params, **dict(zip(parameter_names, theta_original)))

    # Simulate with updated parameters
    results_sim = engine_simulation(updated_params)
    pred_x = np.array([results_sim[key] for key in keys], dtype=float)

    # Compute MSE between normalized predicted and observed outputs
    return np.mean((pred_x - observed_data) ** 2)

# Solve the inverse problem using scipy.optimize
def solve_inverse_problem(engine_simulation, engine_params,
                          observed_data, parameter_names, keys, theta_nominal, margin=0.05):
    """
    Solves the inverse problem by optimizing parameters to minimize the difference
    between observed and simulated outputs.

    Args:
        engine_simulation: The simulation function.
        engine_params: Initial engine parameters dataclass.
        observed_data: Observed output data.
        parameter_names: List of parameter names to optimize.
        keys: Observable output variable keys.
        nominal_theta: Nominal values for theta parameters.
        margin: Margin for min-max normalization of theta.

    Returns:
        A tuple containing optimized parameters, optimized outputs, and the optimization result.
    """
    # Normalize theta
    theta_min = theta_nominal * (1 - margin)
    theta_max = theta_nominal * (1 + margin)
    theta_normalized = min_max_normalize(theta_nominal, theta_min, theta_max)

    # Adjust lower and upper bounds based on parameter types
    lower_bounds = []
    upper_bounds = []

    for name in parameter_names:
        if 'efficiency' in name:
            lower_bounds.append(0.8)  # Efficiencies between 0.7 and 1.0
            upper_bounds.append(1.0)
        elif 'PR' in name:
            lower_bounds.append(1.0)  # Pressure ratios must be > 1
            upper_bounds.append(50.0)  # Adjust upper limit as needed
        else:
            lower_bounds.append(0.0)  # Adjust as necessary
            upper_bounds.append(np.inf)

    # Normalize bounds for optimization
    normalized_bounds = normalize_bounds(
        np.array(lower_bounds), np.array(upper_bounds), theta_min, theta_max
    )

    # Run optimization
    result = minimize(
        fun=loss_function,
        x0=theta_normalized,
        args=(engine_simulation, engine_params,
              observed_data, parameter_names, keys, theta_min, theta_max),
        method='L-BFGS-B',
        bounds=normalized_bounds,  # Use normalized bounds
        options={'disp': True, 'maxiter': 500, 'eps': 1e-2}
    )

    # Extract optimized theta and denormalize
    theta_optimized_normalized = result.x
    theta_optimized = min_max_denormalize(theta_optimized_normalized, theta_min, theta_max)

    # Update engine parameters with optimized values
    optimized_params = replace(engine_params, **dict(zip(parameter_names, theta_optimized)))

    # Simulate with optimized parameters
    results_optimized = engine_simulation(optimized_params)

    # Extract optimized outputs
    X_optimized = np.array([results_optimized[key] for key in keys], dtype=float)

    # Print detailed optimization results
    print("\nOptimization Results:")
    print(f"Optimized Parameters (Theta): {result.x}")
    print(f"Final Loss: {result.fun}")
    print(f"Number of Iterations: {result.nit}")
    print(f"Number of Function Evaluations: {result.nfev}")
    print(f"Convergence Message: {result.message}")

    return theta_optimized


# Example usage
if __name__ == "__main__":
    
    # Define observed outputs (from book)
    observed = {
        "T2": 244, "P2": 34.3, "W2": 100,
        "T24": 301, "P24": 65.17, "W24": 18.18,
        "W13": 81.82,"P17": 63.21, "P26": 63.87,
        "T27": 505.1,
        "T3": 709.2, "P3": 894.2, "W3": 17.91,
        "P31": 876.3, "W31": 15.09,
        "T41": 1400, "P41": 850, "W41": 16.78,
        "WF": 0.326, "T4": 1468.8, "W4": 15.42,
        "T415": 1362.4, "W415": 17.63,
        "T44": 995.4,  "P44": 207.31, "W44": 17.63,
        "P46": 205.22, "T48": 713.4, "P48": 45.80,
        "W5": 18.41, "T5": 707.7, "P7": 45.34
        }

    # Define keys based on available data
    keys = list(observed.keys())

    # Convert observed data to arrays
    observed_data = np.array([observed[key] for key in keys], dtype=float)
    
    # Define parameter names to be optimized
    parameter_names = [
        #'T_amb', 'P_amb',
        #'mach_number',
        #'mass_flow', 'SOT', 'bypass_ratio',
        'fan_PR',
        'LP_compressor_PR',
        'HP_compressor_PR',
        'fan_efficiency', 'LP_compressor_efficiency', 'HP_compressor_efficiency',
        'LP_turbine_efficiency', 'HP_turbine_efficiency',
        #'intake_pressure_loss', 'bypass_duct_pressure_loss', 'inter_compressor_pressure_loss',
        #'compressor_exit_diffuser_pressure_loss', 'combustor_pressure_loss', 'inter_turbine_duct_pressure_loss',
        #'jet_pipe_pressure_loss', 'bleed_extraction', 'cooling_mechanical_loss',
        'TURB_cooling_flow_percentage', 'COMB_cooling_flow_percentage',
        'HPT_cooling_flow_percentage',  'LPT_cooling_flow_percentage'
    ]

    # Initial values for tunable parameters
    engine_params = EngineParameters()  # Default parameters
    tunable_initial = np.array([getattr(engine_params, name) for name in parameter_names])*1.005

    # Solve the inverse problem
    optimized_theta = solve_inverse_problem(engine_simulation, engine_params,
                          observed_data, parameter_names, keys, tunable_initial)

    # Evaluate results
    optimized_params = replace(engine_params, **dict(zip(parameter_names, optimized_theta)))
    results_optimized = engine_simulation(optimized_params)
    print("Estimated Theta:", optimized_theta)

    # Plot results
    X_keys = keys
    X_observed = observed_data
    X_optimized = np.array([results_optimized[key] for key in keys], dtype=float)

    # DataFrames for comparing true vs. optimized values
    df_X_comparison = pd.DataFrame({
        'Observable Variable (X)': X_keys,
        'Obs Value': X_observed,
        'Optimized Value': X_optimized,
        'Difference': np.abs(X_observed - X_optimized)
    })
    print("\nComparison of Observable Variables (X):")
    print(df_X_comparison)

    # Create a DataFrame to display these values side by side
    df_tunable_comparison = pd.DataFrame({
        'Parameter': parameter_names,
        'Optimized Value': optimized_theta,
        'Initial Value': tunable_initial
    })

    # Display the DataFrame
    print('')
    print("Comparison of True, Initial, and Optimized Values for Tunable Parameters")
    print(df_tunable_comparison)

