import math
import os
from dataclasses import dataclass
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.utils import subgraph

# Step 1: Load CMAPSS Data from the HDF5 File
file_path = 'N-CMAPSS_DS01-005.h5'

with h5py.File(file_path, 'r') as hdf:
    # Extract train and test sets from CMAPSS data
    W_train = np.array(hdf.get('W_dev'))  # Operational conditions
    Xs_train = np.array(hdf.get('X_s_dev'))  # Sensor readings
    RUL_train = np.array(hdf.get('Y_dev'))  # Remaining Useful Life (RUL)
    A_train = np.array(hdf.get('A_dev'))  # Units and Cycles

    W_test = np.array(hdf.get('W_test'))
    Xs_test = np.array(hdf.get('X_s_test'))
    RUL_test = np.array(hdf.get('Y_test'))
    A_test = np.array(hdf.get('A_test'))

#define the device to run the code on the GPRO cluster (ZHAW server)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Table of Nodes
# --------------
# Index   Sensor    Description	                       Units

# 0	     T24	   Total temperature at LPC outlet	   °R
# 1	     T30	   Total temperature at HPC outlet	   °R
# 2	     T48	   Total temperature at HPT outlet	   °R
# 3	     T50	   Total temperature at LPT outlet	   °R
# 4	     P15	   Total pressure in bypass-duct	   psia
# 5	     P2	       Total pressure at fan inlet	       psia
# 6	     P21	   Total pressure at fan outlet	       psia
# 7	     P24	   Total pressure at LPC outlet	       psia
# 8	     Ps30	   Static pressure at HPC outlet	   psia
# 9	     P40	   Total pressure at burner outlet	   psia
# 10	 P50	   Total pressure at LPT outlet	       psia
# 11	 Nf	       Physical fan speed	               rpm
# 12	 Nc	       Physical core speed	               rpm
# 13	 Wf	       Fuel flow	                       pps
# 14     alt       Altitude                            ft
# 15     Mach      Flight Mach number                  -
# 16     TRA       Throttle-resolver angle             %
# 17     T2        Total temperature at fan inlet      °R

#--------------


# Downsample the data by taking every 10th sample
W_train = W_train[::10]
Xs_train = Xs_train[::10]
RUL_train = RUL_train[::10]
A_train = A_train[::10]

W_test = W_test[::10]
Xs_test = Xs_test[::10]
RUL_test = RUL_test[::10]
A_test = A_test[::10]


# Normalize the sensor and operational data
sensor_scaler = MinMaxScaler(feature_range=(0, 1)) # Normalize to range [0, 1]
W_scaler = MinMaxScaler(feature_range=(0, 1))

Xs_train = sensor_scaler.fit_transform(Xs_train[:, :14])  # Only first 14 columns (sensor nodes)
W_train = W_scaler.fit_transform(W_train)  # Keep the operational node normalization separate if needed

# Apply the same transformation to test data
Xs_test = sensor_scaler.transform(Xs_test[:, :14])
W_test = W_scaler.transform(W_test)

# A list of all sensor and operational condition variable names (how they appear in combined_data)
X_var_W_var = ["T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24", "Ps30", "P40", "P50",
               "Nf", "Nc", "Wf", "alt", "Mach", "TRA", "T2"]

# A dictionary that maps each variable name to its index in combined_data
mapping_var_colX = {}
node_counter = 0
for var in X_var_W_var:
    # Assign the current index to the variable and update the counter (e.g T24 assign to 0, T30 belongs to 1 etc.)
    mapping_var_colX[var] = node_counter
    # Increment the counter for the next variable of this type
    node_counter += 1


# the list of all node pairs (edges) in the graph (23)
mappings = [
    ["T2", "T24"],  # T24 depends on T2
    # Equation: T24 = (T2 / ETA2) * (fan_PR ** ((gamma - 1) / gamma) - 1) + T2

    ["T24", "T30"],  # T30 depends on T24
    # Equation: T27 = (T26 / ETA_LPC) * (LP_compressor_PR ** ((gamma - 1) / gamma) - 1) + T26
    # T27 depends on T26 (which is T24), and T3 (T30) depends on T27

    ["T30", "T48"],  # T48 depends on T30
    # Equations:
    # T31 = T3  # T31 depends on T30
    # FAR = compute_far(T31, T41, combustion_efficiency)
    # WF depends on FAR and affects T415 and T416 (which is T44, corresponding to T48)

    ["T48", "T50"],  # T50 depends on T48
    # Equation: T48 (which corresponds to T50) depends on T46 and PW46, which in turn depend on T48

    ["P2", "P21"],   # P21 depends on P2
    # Equation: P13 = fan_PR * P2; P21 corresponds to P13

    ["P21", "P24"],  # P24 depends on P21
    # Equation: P27 = LP_compressor_PR * P26; P26 = P24 * (1 - inter_compressor_pressure_loss); P24 = P13 (P21)

    ["P24", "Ps30"],  # Ps30 depends on P24
    # Equation: P3 = HP_compressor_PR * P27; Ps30 corresponds to P3

    ["Ps30", "T30"],  # T30 depends on Ps30
    # Equation: T3 (T30) is calculated after P3 (Ps30)

    ["Ps30", "P40"],  # P40 depends on Ps30
    # Equations:
    # P31 = P3 * (1 - compressor_exit_diffuser_pressure_loss)
    # P41 = P31 * (1 - combustor_pressure_loss); P40 corresponds to P41

    ["P40", "P50"],  # P50 depends on P40
    # Equation: P7 (P50) depends on P48, which is influenced by P41 (P40)

    ["T30", "Wf"],   # Wf depends on T30
    # Equation: FAR = compute_far(T31, T41, combustion_efficiency); T31 = T3 (T30); WF depends on FAR

    ["Wf", "T48"],   # T48 depends on Wf
    # Equations:
    # WF affects W41 and T4, which influence T415 and T416 (T48)

    ["Wf", "P40"],   # P40 depends on Wf
    # Equation: WF affects combustion pressure losses, influencing P41 (P40)

    ["P2", "T24"],   # T24 depends on P2
    # Equations:
    # P2 affects P13 (P21), which influences P24 and indirectly affects T24 through pressure ratios

    ["T2", "P2"],    # T2 depends on P2
    # Equations:
    # T2 and P2 are both derived from T0 and P0, linking them together

    ["Mach", "T2"],  # T2 depends on Mach
    # Equation: T0 = (1 + ((gamma - 1)/2) * Mach^2) * T_amb; T2 = T0

    ["Mach", "P2"],  # P2 depends on Mach
    # Equation: P0 = ((T0 / T_amb)^(gamma / (gamma - 1))) * P_amb; P2 = P0 * (1 - intake_pressure_loss)

    ["alt", "P2"],   # P2 depends on altitude
    # Equation: P_amb (ambient pressure) depends on altitude; P2 is influenced by P_amb

    ["TRA", "Wf"],   # Wf depends on TRA
    # Assumption: Throttle-resolver angle (TRA) affects fuel flow (Wf)

    ["Nc", "T48"],   # T48 depends on Nc
    # Assumption: Core speed (Nc) affects HP turbine temperature (T48) due to work extraction

    ["Nc", "T30"],   # T30 depends on Nc
    # Assumption: Core speed (Nc) influences HPC outlet temperature (T30) through compressor work

    ["Nf", "T50"],   # T50 depends on Nf
    # Assumption: Fan speed (Nf) affects LPT outlet temperature (T50) via load changes

    ["Nf", "P21"],   # P21 depends on Nf
    # Assumption: Fan speed (Nf) influences fan outlet pressure (P21)
]

# (17)
bidirectional_edges = [
    ["T24", "T30"],
    ["T30", "T48"],
    ["T48", "T50"],
    ["P2", "P21"],
    ["P21", "P24"],
    ["P24", "Ps30"],
    ["Ps30", "T30"],
    ["Ps30", "P40"],
    ["P40", "P50"],
    ["T30", "Wf"],
    ["Wf", "T48"],
    ["Wf", "P40"],
    ["P2", "T24"],
    ["Nc", "T48"],
    ["Nc", "T30"],
    ["Nf", "T50"],
    ["Nf", "P21"]
]

def create_edges(mapping, mapping_var_colX, bidirectional=False):
    edges = []
    for pair in mapping:
        edges.append((mapping_var_colX[pair[0]], mapping_var_colX[pair[1]]))
        if bidirectional:
            edges.append((mapping_var_colX[pair[1]], mapping_var_colX[pair[0]]))
    return edges

def get_edges_for_type(mappings, mapping_var_colX, bidirectional_edges):
    edges = []
    for pair in mappings:
        # Check if the pair should be bidirectional
        bidirectional = pair in bidirectional_edges or pair[::-1] in bidirectional_edges
        # Create edges based on bidirectionality
        edges.extend(create_edges([pair], mapping_var_colX, bidirectional=bidirectional))
    return edges


#Prepare Data By Cycle (after concatenation)
def prepare_data_by_cycle(combined_data, cycle_indices):
    #Organizes concatenated sensor and operational data by each cycle.
    #Returns a dictionary with each cycle's concatenated data as a separate entry.
    data_by_cycle = {}

    # Loop through each unique cycle
    for cycle in np.unique(cycle_indices):
        # Get the mask for the current cycle
        cycle_mask = cycle_indices == cycle
        # Select combined data for the current cycle
        data_by_cycle[cycle] = combined_data[cycle_mask]

    return data_by_cycle

# relevant constants for the GE90 Engine
@dataclass
class EngineParameters:
    cp: float = 1.005
    cp_turbine: float = 1.15
    gamma: float = 1.4
    gamma_turbine: float = 1.333333333333
    T_amb: float = 216.7  # K
    P_amb: float = 22.628  # kPa
    mach_number: float = 0.61
    SOT: float = 1480  # K
    mass_flow: float = 812  # kg/s
    bypass_ratio: float = 14.5
    fan_PR: float = 1.9
    LP_compressor_PR: float = 2.0
    HP_compressor_PR: float = 18.0
    fan_efficiency: float = 0.92
    LP_compressor_efficiency: float = 0.92
    HP_compressor_efficiency: float = 0.92
    LP_turbine_efficiency: float = 0.89
    HP_turbine_efficiency: float = 0.90
    combustion_efficiency: float = 0.999
    mechanical_efficiency: float = 0.995
    intake_pressure_loss: float = 0.005
    bypass_duct_pressure_loss: float = 0.03
    inter_compressor_pressure_loss: float = 0.02
    compressor_exit_diffuser_pressure_loss: float = 0.02
    combustor_pressure_loss: float = 0.03
    inter_turbine_duct_pressure_loss: float = 0.01
    jet_pipe_pressure_loss: float = 0.01
    bleed_extraction: float = 0.01
    cooling_mechanical_loss: float = 0.005
    COMB_cooling_flow_percentage: float = 0.075
    TURB_cooling_flow_percentage: float = 0.08
    HPT_cooling_flow_percentage: float = 0.047
    LPT_cooling_flow_percentage: float = 0.033

#processing the dataset using the sliding window approach to prepare graphs for the model
def create_graphs_with_sliding_window(combined_data, cycle, unit, window_size=50):
    data_list = []
    params = EngineParameters()

    # Thermodynamic helper functions
    def compute_eta(PR, gamma, efficiency):
        numerator = PR ** ((gamma - 1) / gamma) - 1
        denominator = PR ** ((gamma - 1) / (gamma * efficiency)) - 1
        ETA = numerator / denominator if denominator != 0 else np.finfo(float).eps
        return ETA

    def compute_eta_from_Temps(T_in, T_out, PR, gamma):
        return (T_out / T_in - 1) / (PR ** ((gamma - 1) / gamma) - 1)

    # Necessary functions for all edge pairs
    def calc_T2_to_T24(T2, fan_PR, gamma, fan_efficiency):
        ETA2 = compute_eta(fan_PR, gamma, fan_efficiency)
        PR_exponent = (gamma - 1) / gamma
        A = fan_PR ** PR_exponent - 1
        T24 = (T2 / ETA2) * A + T2
        return T24

    def calc_P40_to_P50(P40, turbine_pressure_losses):
        P50 = P40 * (1 - turbine_pressure_losses)
        return P50

    def calc_P50_to_P40(P50, turbine_pressure_losses):
        P40 = P50 / (1 - turbine_pressure_losses)
        return P40

    def calc_T24_to_T30(T24, LP_compressor_PR, HP_compressor_PR, gamma,
                        LP_compressor_efficiency, HP_compressor_efficiency):
        # LPC Stage
        ETA_LPC = compute_eta(LP_compressor_PR, gamma, LP_compressor_efficiency)
        PR_exp = (gamma - 1) / gamma
        A_LPC = LP_compressor_PR ** PR_exp - 1
        T27 = (T24 / ETA_LPC) * A_LPC + T24

        # HPC Stage
        ETA_HPC = compute_eta(HP_compressor_PR, gamma, HP_compressor_efficiency)
        A_HPC = HP_compressor_PR ** PR_exp - 1
        T30 = (T27 / ETA_HPC) * A_HPC + T27
        return T30

    def calc_T30_to_T24(T30, LP_compressor_PR, HP_compressor_PR, gamma,
                        LP_compressor_efficiency, HP_compressor_efficiency):
        # HPC Stage (inverse)
        ETA_HPC = compute_eta(HP_compressor_PR, gamma, HP_compressor_efficiency)
        PR_exp = (gamma - 1) / gamma
        A_HPC = HP_compressor_PR ** PR_exp - 1
        T27 = T30 / (1 + (A_HPC / ETA_HPC))

        # LPC Stage (inverse)
        ETA_LPC = compute_eta(LP_compressor_PR, gamma, LP_compressor_efficiency)
        A_LPC = LP_compressor_PR ** PR_exp - 1
        T24 = T27 / (1 + (A_LPC / ETA_LPC))
        return T24

    def calc_P2_to_P21(P2, fan_PR):
        return fan_PR * P2

    def calc_P21_to_P2(P21, fan_PR):
        return P21 / fan_PR

    def calc_P21_to_P24(P21):
        # Assuming no loss
        return P21

    def calc_P24_to_P21(P24):
        # Assuming no loss
        return P24

    def calc_P24_to_Ps30(P24, LP_compressor_PR, HP_compressor_PR, inter_compressor_pressure_loss):
        P26 = P24 * (1 - inter_compressor_pressure_loss)
        PR_exp = (gamma - 1) / gamma
        # P27 = LP compressor exit
        P27 = LP_compressor_PR * P26
        # Ps30 = HPC outlet = HP_compressor_PR * P27
        Ps30 = HP_compressor_PR * P27
        return Ps30

    def calc_Ps30_to_P24(Ps30, LP_compressor_PR, HP_compressor_PR, inter_compressor_pressure_loss):
        P27 = Ps30 / HP_compressor_PR
        P26 = P27 / LP_compressor_PR
        P24 = P26 / (1 - inter_compressor_pressure_loss)
        return P24

    def calc_Ps30_to_T30(T27, HP_compressor_PR, gamma, HP_compressor_efficiency):
        ETA_HPC = compute_eta(HP_compressor_PR, gamma, HP_compressor_efficiency)
        PR_exp = (gamma - 1) / gamma
        A_HPC = HP_compressor_PR ** PR_exp - 1
        T30 = (T27 / ETA_HPC) * A_HPC + T27
        return T30

    def calc_T30_to_Ps30(T30, T27, gamma, HP_compressor_efficiency):
        # Not directly invertible; return a placeholder (used in edges)
        return 1.0

    def calc_T2_to_P2(T2, Mach, gamma, T_amb, P_amb, intake_pressure_loss):
        T0 = T2  # Because we considered T2 ~ T0
        P0 = ((T0 / T_amb) ** (gamma / (gamma - 1))) * P_amb
        P2 = P0 * (1 - intake_pressure_loss)
        return P2

    def calc_Mach_to_T2(Mach, gamma, T_amb):
        return (1 + ((gamma - 1) / 2) * Mach ** 2) * T_amb

    def calc_Mach_to_P2(Mach, gamma, T_amb, P_amb, intake_pressure_loss):
        T2 = calc_Mach_to_T2(Mach, gamma, T_amb)
        P0 = ((T2 / T_amb) ** (gamma / (gamma - 1))) * P_amb
        P2 = P0 * (1 - intake_pressure_loss)
        return P2

    def calc_ratio_based(value, prev_value_target, prev_value_source):
        if prev_value_source != 0:
            ratio = prev_value_target / prev_value_source
            return ratio * value
        return 0

    def calc_ratio_inverse(value, prev_value_target, prev_value_source):
        if prev_value_source != 0:
            ratio = prev_value_target / prev_value_source
            if ratio != 0:
                return value / ratio
        return 0

    def calc_Nc_to_T30(Nc, Nc_last, T30_last):
        return calc_ratio_based(Nc, T30_last, Nc_last)

    def calc_T30_to_Nc(T30, T30_last, Nc_last):
        return calc_ratio_inverse(T30, T30_last, Nc_last)

    def calc_Nf_to_P21(Nf, Nf_last, P21_last):
        return calc_ratio_based(Nf, P21_last, Nf_last)

    def calc_P21_to_Nf(P21, P21_last, Nf_last):
        return calc_ratio_inverse(P21, P21_last, Nf_last)

    def calc_TRA_to_Wf(TRA, TRA_last, Wf_last):
        return calc_ratio_based(TRA, Wf_last, TRA_last)

    def calc_Nc_to_T48(Nc, Nc_last, T48_last):
        return calc_ratio_based(Nc, T48_last, Nc_last)

    def calc_T48_to_Nc(T48, T48_last, Nc_last):
        return calc_ratio_inverse(T48, T48_last, Nc_last)


    for start_row in range(len(combined_data) - window_size):
        window_data = combined_data[start_row:start_row + window_size]
        x = torch.tensor(window_data.T, dtype=torch.float32)

        target_row = combined_data[start_row + window_size]
        y = torch.tensor(target_row.T, dtype=torch.float32)

        edge_index = torch.tensor(
            get_edges_for_type(mappings, mapping_var_colX, bidirectional_edges), dtype=torch.long
        ).t().contiguous()

        if x.shape[1] > 1:
            x_prev = x[:, -2]
        else:
            x_prev = x[:, -1]

        variable_indices = {
            'T24': 0,
            'T30': 1,
            'T48': 2,
            'T50': 3,
            'P15': 4,
            'P2': 5,
            'P21': 6,
            'P24': 7,
            'Ps30':8,
            'P40':9,
            'P50':10,
            'Nf':11,
            'Nc':12,
            'Wf':13,
            'alt':14,
            'Mach':15,
            'TRA':16,
            'T2':17
        }

        x_last = x[:, -1]
        T2 = x_last[variable_indices['T2']].item()
        T24 = x_last[variable_indices['T24']].item()
        T30 = x_last[variable_indices['T30']].item()
        T48 = x_last[variable_indices['T48']].item()
        T50 = x_last[variable_indices['T50']].item()
        P2 = x_last[variable_indices['P2']].item()
        P21 = x_last[variable_indices['P21']].item()
        P24 = x_last[variable_indices['P24']].item()
        Ps30 = x_last[variable_indices['Ps30']].item()
        P40 = x_last[variable_indices['P40']].item()
        P50 = x_last[variable_indices['P50']].item()
        Nf = x_last[variable_indices['Nf']].item()
        Nc = x_last[variable_indices['Nc']].item()
        Wf = x_last[variable_indices['Wf']].item()
        alt = x_last[variable_indices['alt']].item()
        Mach = x_last[variable_indices['Mach']].item()
        TRA = x_last[variable_indices['TRA']].item()
        P15 = x_last[variable_indices['P15']].item()

        Nc_last = x_prev[variable_indices['Nc']].item()
        T30_last = x_prev[variable_indices['T30']].item()
        Nf_last = x_prev[variable_indices['Nf']].item()
        P21_last = x_prev[variable_indices['P21']].item()
        TRA_last = x_prev[variable_indices['TRA']].item()
        Wf_last = x_prev[variable_indices['Wf']].item()
        T48_last = x_prev[variable_indices['T48']].item()

        gamma = params.gamma
        cp = params.cp
        fan_efficiency = params.fan_efficiency
        LP_compressor_efficiency = params.LP_compressor_efficiency
        HP_compressor_efficiency = params.HP_compressor_efficiency
        fan_PR = params.fan_PR
        LP_compressor_PR = params.LP_compressor_PR
        HP_compressor_PR = params.HP_compressor_PR
        inter_compressor_pressure_loss = params.inter_compressor_pressure_loss
        intake_pressure_loss = params.intake_pressure_loss
        P_amb = params.P_amb
        T_amb = params.T_amb
        SOT = params.SOT
        turbine_pressure_losses = params.inter_turbine_duct_pressure_loss + params.jet_pipe_pressure_loss

        # Before calling calc_Ps30_to_T30 and calc_T30_to_Ps30 we need T27:
        ETA_LPC = compute_eta(LP_compressor_PR, gamma, LP_compressor_efficiency)
        PR_exp = (gamma - 1) / gamma
        A_LPC = LP_compressor_PR ** PR_exp - 1
        T27 = (T24 / ETA_LPC) * A_LPC + T24  # HPC inlet temperature

        edge_weights_list = []

        # 0: ("T2", "T24")
        ew = calc_T2_to_T24(T2, fan_PR, gamma, fan_efficiency)
        edge_weights_list.append(ew)

        # 1: ("T24", "T30")
        ew = calc_T24_to_T30(T24, LP_compressor_PR, HP_compressor_PR, gamma,
                             LP_compressor_efficiency, HP_compressor_efficiency)
        edge_weights_list.append(ew)

        # 2: ("T30", "T24")
        ew = calc_T30_to_T24(T30, LP_compressor_PR, HP_compressor_PR, gamma,
                             LP_compressor_efficiency, HP_compressor_efficiency)
        edge_weights_list.append(ew)

        # 3: ("T30", "T48")
        ew = SOT
        edge_weights_list.append(ew)

        # 4: ("T48", "T30")
        ew = T30
        edge_weights_list.append(ew)

        # 5: ("T48", "T50")
        ew = 1.0
        edge_weights_list.append(ew)

        # 6: ("T50", "T48")
        ew = 1.0
        edge_weights_list.append(ew)

        # 7: ("P2", "P21")
        ew = calc_P2_to_P21(P2, fan_PR)
        edge_weights_list.append(ew)

        # 8: ("P21", "P2")
        ew = calc_P21_to_P2(P21, fan_PR)
        edge_weights_list.append(ew)

        # 9: ("P21", "P24")
        ew = calc_P21_to_P24(P21)
        edge_weights_list.append(ew)

        # 10: ("P24", "P21")
        ew = calc_P24_to_P21(P24)
        edge_weights_list.append(ew)

        # 11: ("P24", "Ps30")
        ew = calc_P24_to_Ps30(P24, LP_compressor_PR, HP_compressor_PR, inter_compressor_pressure_loss)
        edge_weights_list.append(ew)

        # 12: ("Ps30", "P24")
        ew = calc_Ps30_to_P24(Ps30, LP_compressor_PR, HP_compressor_PR, inter_compressor_pressure_loss)
        edge_weights_list.append(ew)

        # 13: ("Ps30", "T30")
        ew = calc_Ps30_to_T30(T27, HP_compressor_PR, gamma, HP_compressor_efficiency)
        edge_weights_list.append(ew)

        # 14: ("T30", "Ps30")
        ew = calc_T30_to_Ps30(T30, T27, gamma, HP_compressor_efficiency)
        edge_weights_list.append(ew)

        # 15: ("Ps30", "P40")
        ew = 1.0
        edge_weights_list.append(ew)

        # 16: ("P40", "Ps30")
        ew = 1.0
        edge_weights_list.append(ew)

        # 17: ("P40", "P50")
        ew = calc_P40_to_P50(P40, turbine_pressure_losses)
        edge_weights_list.append(ew)

        # 18: ("P50", "P40")
        ew = calc_P50_to_P40(P50, turbine_pressure_losses)
        edge_weights_list.append(ew)

        # 19: ("T30", "Wf")
        ew = 1.0
        edge_weights_list.append(ew)

        # 20: ("Wf", "T30")
        ew = 1.0
        edge_weights_list.append(ew)

        # 21: ("Wf", "T48")
        ew = 1.0
        edge_weights_list.append(ew)

        # 22: ("T48", "Wf")
        ew = 1.0
        edge_weights_list.append(ew)

        # 23: ("Wf", "P40")
        ew = 1.0
        edge_weights_list.append(ew)

        # 24: ("P40", "Wf")
        ew = 1.0
        edge_weights_list.append(ew)

        # 25: ("P2", "T24")
        ew = 1.0
        edge_weights_list.append(ew)

        # 26: ("T24", "P2")
        ew = 1.0
        edge_weights_list.append(ew)

        # 27: ("T2", "P2")
        ew = calc_T2_to_P2(T2, Mach, gamma, T_amb, P_amb, intake_pressure_loss)
        edge_weights_list.append(ew)

        # 28: ("Mach", "T2")
        ew = calc_Mach_to_T2(Mach, gamma, T_amb)
        edge_weights_list.append(ew)

        # 29: ("Mach", "P2")
        ew = calc_Mach_to_P2(Mach, gamma, T_amb, P_amb, intake_pressure_loss)
        edge_weights_list.append(ew)

        # 30: ("alt", "P2")
        ew = 1.0
        edge_weights_list.append(ew)

        # 31: ("TRA", "Wf") - ratio
        Wf_calc = calc_TRA_to_Wf(TRA, TRA_last, Wf_last)
        ew = abs(Wf - Wf_calc)
        edge_weights_list.append(ew)

        # 32: ("Nc", "T48") - ratio
        T48_calc = calc_Nc_to_T48(Nc, Nc_last, T48_last)
        ew = abs(T48 - T48_calc)
        edge_weights_list.append(ew)

        # 33: ("T48", "Nc") - inverse ratio
        Nc_calc = calc_T48_to_Nc(T48, T48_last, Nc_last)
        ew = abs(Nc - Nc_calc)
        edge_weights_list.append(ew)

        # 34: ("Nc", "T30") - ratio
        T30_calc = calc_Nc_to_T30(Nc, Nc_last, T30_last)
        ew = abs(T30 - T30_calc)
        edge_weights_list.append(ew)

        # 35: ("T30", "Nc") - inverse ratio
        Nc_calc = calc_T30_to_Nc(T30, T30_last, Nc_last)
        ew = abs(Nc - Nc_calc)
        edge_weights_list.append(ew)

        # 36: ("Nf", "T50")
        ew = 1.0
        edge_weights_list.append(ew)

        # 37: ("T50", "Nf")
        ew = 1.0
        edge_weights_list.append(ew)

        # 38: ("Nf", "P21") - ratio
        P21_calc = calc_Nf_to_P21(Nf, Nf_last, P21_last)
        ew = abs(P21 - P21_calc)
        edge_weights_list.append(ew)

        # 39: ("P21", "Nf") - inverse ratio
        Nf_calc = calc_P21_to_Nf(P21, P21_last, Nf_last)
        ew = abs(Nf - Nf_calc)
        edge_weights_list.append(ew)

        edge_weights_tensor = torch.tensor(edge_weights_list, dtype=torch.float32)

        # Compute min and max
        min_val = torch.min(edge_weights_tensor)
        max_val = torch.max(edge_weights_tensor)

        # Apply min-max normalization if there's a range:
        if max_val > min_val:
            edge_weights_tensor = (edge_weights_tensor - min_val) / (max_val - min_val)
        else:
            # If all values are the same, just set them to one
            edge_weights_tensor = torch.ones_like(edge_weights_tensor)

        # Now edge_weights_tensor is on a [0,1] scale
        edge_weights = edge_weights_tensor

        #create the graph from the window with the edges and edge weights, along with their corresponding cycle and unit
        graph = Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y)
        graph.cycle = torch.tensor([cycle], dtype=torch.int32)
        graph.unit = torch.tensor([unit], dtype=torch.int32)
        data_list.append(graph)

    return data_list


# Step 2: Create Data Lists and call the sliding window function
window_size = 50  # Define the window size

# For training data
unique_units_train = np.unique(A_train[:, 0]) # Get a list of unique Units (6 training units)
train_data_list = []

for unit in unique_units_train:
    # Get indices for the current unit
    unit_indices = np.where(A_train[:, 0] == unit)[0]

    # Get all its cycles for the current unit
    unit_cycles = A_train[unit_indices, 1]

    # Mask for cycles <= 20
    unit_cycle_mask = unit_cycles <= 20

    # From the list of all cycles of the current unit, get me the indices of the first 20 cycles (only data of healthy state)
    selected_indices = unit_indices[unit_cycle_mask]

    sensor_data_unit = Xs_train[selected_indices]
    w_data_unit = W_train[selected_indices]

    # Concatenate sensor data (X) with operational conditions (W) once
    combined_data_unit = np.concatenate((sensor_data_unit, w_data_unit), axis=1)

    cycle_indices = A_train[selected_indices, 1]
    # Organize data by cycle before calling the sliding window function
    data_by_cycle = prepare_data_by_cycle(combined_data_unit, cycle_indices)

    for cycle, single_cycle_data in data_by_cycle.items():
        # Check if there are enough samples for the sliding window
        if len(single_cycle_data) > window_size:
            # Create graphs for the training data of 50 rows
            data_list_unit = create_graphs_with_sliding_window(single_cycle_data, cycle, unit, window_size=window_size)
            train_data_list.extend(data_list_unit)
        else:
            print(f"Training data has insufficient samples for window size {window_size}")

# For testing data
unique_units_test = np.unique(A_test[:,0])
test_data_list = []

for unit in unique_units_test:
    # Get indices for the current unit
    unit_indices = np.where(A_test[:, 0] == unit)[0]

    # Get all cycles for the current unit
    unit_cycles = A_test[unit_indices, 1]

    # **No cycle masking needed** since I want all cycles for test units
    selected_indices = unit_indices  # Use all cycles

    # Sensor and operational data for the current unit (normalized)
    sensor_data_unit = Xs_test[selected_indices]
    w_data_unit = W_test[selected_indices]

    # Concatenate sensor data (X) with operational conditions (W) once
    combined_data_unit = np.concatenate((sensor_data_unit, w_data_unit), axis=1)

    cycle_indices = A_test[selected_indices, 1]
    # Organize data by cycle before calling the sliding window function
    data_by_cycle = prepare_data_by_cycle(combined_data_unit, cycle_indices)

    for cycle, single_cycle_data in data_by_cycle.items():
        if len(single_cycle_data) > window_size:
            # Create graphs for the test data
            data_list_unit = create_graphs_with_sliding_window(single_cycle_data, cycle, unit, window_size=window_size)
            test_data_list.extend(data_list_unit)
        else:
            print(f"Test data has insufficient samples for window size {window_size}")


#Do windowing over ALL cycles from all training units (for evaluating not only test units but also train units)
unique_units_train_all_cycles = np.unique(A_train[:,0])
train_all_cycles_list = []

for unit in unique_units_train_all_cycles:
    # Get indices for the current unit
    unit_indices = np.where(A_train[:, 0] == unit)[0]

    # Get all cycles for the current unit
    unit_cycles = A_train[unit_indices, 1]

    selected_indices = unit_indices  # Use all cycles

    # Sensor and operational data for the current unit (normalized)
    sensor_data_unit = Xs_train[selected_indices]
    w_data_unit = W_train[selected_indices]

    # Concatenate sensor data (X) with operational conditions (W) once
    combined_data_unit = np.concatenate((sensor_data_unit, w_data_unit), axis=1)

    cycle_indices = A_train[selected_indices, 1]
    # Organize data by cycle before calling the sliding window function
    data_by_cycle = prepare_data_by_cycle(combined_data_unit, cycle_indices)

    for cycle, single_cycle_data in data_by_cycle.items():
        if len(single_cycle_data) > window_size:
            # Create graphs for the test data
            data_list_unit_for_all_cycles = create_graphs_with_sliding_window(single_cycle_data, cycle, unit, window_size=window_size)
            train_all_cycles_list.extend(data_list_unit_for_all_cycles)
        else:
            print(f"Test data has insufficient samples for window size {window_size}")


# Step 3: Create DataLoaders
train_loader = DataLoader(train_data_list, batch_size=1024, shuffle=False)
#second train loader with train data list with all cycles
train_loader_all_cycles = DataLoader(train_all_cycles_list, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_data_list, batch_size=1024, shuffle=False)

# Step 4: Define the GNN Model
class GNNSensorsAndConditions(torch.nn.Module):
    def __init__(self):
        super(GNNSensorsAndConditions, self).__init__()
        # Define GCN layers and how the complexity of the feature vector of each node
        self.conv1 = GCNConv(50, 128)
        self.conv2 = GCNConv(128, 64)
        #The dropout rate of 0.2 means that 20% of neurons (or their connections/features) are randomly
        #disabled during training, effectively setting their contributions to zero for that iteration to prevent overfitting and improve the model's generalization.
        self.dropout = torch.nn.Dropout(p=0.2)

        # Separate regression layers for each sensor
        self.sensor_regression_layers = torch.nn.ModuleList(
            [torch.nn.Linear(64, 1) for _ in range(14)])  # 14 nodes as output

    def forward(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x, edge_index, batch, edge_weight = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.edge_weight.to(device)

        #first layer
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        batch_size = data.num_graphs
        num_sensor_nodes = 14
        num_operational_nodes = 4
        total_nodes_per_graph = num_sensor_nodes + num_operational_nodes

        # Total nodes in the batch
        total_nodes_in_batch = x.size(0)

        # Identify sensor nodes across the batch
        is_sensor_node = torch.zeros(total_nodes_in_batch, dtype=torch.bool, device=device)
        for i in range(batch_size):
            start_idx = i * total_nodes_per_graph
            sensor_indices = torch.arange(start_idx, start_idx + num_sensor_nodes, device=device)
            is_sensor_node[sensor_indices] = True

        # Keep only sensor nodes
        sensor_indices = is_sensor_node.nonzero(as_tuple=False).view(-1).to(device)

        # Ensure edge_index and node_mask (sensor_indices) are on the same device
        edge_index = edge_index.to(device)
        sensor_indices = sensor_indices.to(device)

        x = x[sensor_indices]

        # Use subgraph to get the induced subgraph and relabel node indices
        # From 18 nodes to 14 nodes by "kicking out" the operational nodes after the first layer
        edge_index, edge_weight = subgraph(
            sensor_indices,
            edge_index,
            edge_attr=edge_weight,
            relabel_nodes=True,
            num_nodes=total_nodes_in_batch,
        )

        # Update batch information
        batch = batch[sensor_indices]

        #second layer
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # Reshape x to [batch_size, num_sensor_nodes, feature_dim]
        x = x.view(batch_size, num_sensor_nodes, -1)

        # Apply regression layers
        outputs = []
        for node_idx in range(num_sensor_nodes):
            out_node = self.sensor_regression_layers[node_idx](x[:, node_idx, :])  # Shape: [batch_size, 1]
            outputs.append(out_node)

        # Concatenate outputs along the node dimension
        output = torch.cat(outputs, dim=1)  # Shape: [batch_size, num_sensor_nodes]
        output = output.view(-1)  # Flatten if necessary

        return output


# Step 5: Define the Training Loop
def train(model, loader, optimizer, scheduler, criterion, epochs=122):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()  # Zero out the gradients
            predicted = model(batch)  # Forward pass (Shape: [num_sensor_nodes_in_batch, window_size])

            # Dynamically get the batch size and number of total nodes
            batch_size = batch.num_graphs  # Number of graphs in the batch
            total_nodes = batch.y.shape[0] // batch_size  # Total number of nodes per graph
            # Reshape batch.y to [batch_size, total_nodes] and then take only the first 14 sensor nodes
            actual = batch.y.view(batch_size, total_nodes)[:, :14].reshape(-1).to(device)  # Dynamically reshape and flatten

            #Compute the Loss
            loss = criterion(predicted, actual)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the model weights
            total_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(loader):.5f}')


# Step 6: Initialize the Model, Optimizer, and Loss Function
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(device)

# Check which GPU is currently in use (if you have already set a device)
current_device = torch.cuda.current_device()
print(f"Current device: GPU #{current_device}: {torch.cuda.get_device_name(current_device)}")

model = GNNSensorsAndConditions().to(device)  # Instantiate the GNN model and move it to the GPU
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)  # Adam optimizer
criterion = torch.nn.SmoothL1Loss(beta=10).to(device) #Huber loss
# Decrease the learning rate by multiplying the learning rate by 0.5 after every 20 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Train the Model
train(model, train_loader, optimizer, scheduler, criterion, epochs=122)

#Smooth Residuals calculation
def smooth_z_cycle(residual,cycles,units):
    results = []
    stds = []
    cycle_num =[]
    unit_num =[]
    for j in np.unique(units):
        idx = np.ravel(units==j)
        resid_unit = residual[idx, :]
        cycle_unit = cycles[idx]
        for jj in np.unique(cycle_unit):
            idxCycle = np.ravel(cycle_unit ==jj)
            resid_mu_cycle = np.mean(resid_unit[idxCycle],0) #Check this
            std_cycle = np.std(resid_unit[idxCycle],0)
            results.append(resid_mu_cycle)
            stds.append(std_cycle)
            cycle_num.append(jj)
            unit_num.append(j)
    return np.array(results), np.array(stds),np.array(cycle_num),np.array(unit_num)


# Step 7: Define the Evaluation Function
def evaluate(model, loader):
    model.eval()  # Set the model to evaluation mode
    predicted_list = []
    actual_list = []
    num_nodes = 14

    cycles_list = []
    units_list = []

    with torch.no_grad():  # Disable gradient calculation
        for batch in loader:
            batch = batch.to(device)
            predicted = model(batch)  # Get the predictions
            batch_size = batch.num_graphs

            # Ensure that the total number of predictions matches the expected number
            total_nodes = predicted.size(0)
            expected_nodes = batch_size * num_nodes
            assert total_nodes == expected_nodes, f"Expected {expected_nodes} nodes, got {total_nodes}"

            # Reshape predictions to [batch_size, num_nodes]
            predicted = predicted.view(batch_size, num_nodes)
            predicted_np = predicted.cpu().numpy()
            predicted_list.append(predicted_np)

            # Dynamically get the batch size and number of total nodes
            batch_size = batch.num_graphs  # Number of graphs in the batch
            total_nodes = batch.y.shape[0] // batch_size  # Total number of nodes per graph (18)
            # Reshape batch.y to [batch_size, total_nodes] and then take only the first 14 sensor nodes
            actual = batch.y.view(batch_size, total_nodes)[:, :14].reshape(-1).to(device)  # Dynamically reshape and flatten

            # Similarly reshape actual targets
            actual = actual.view(batch_size, num_nodes)
            actual_np = actual.cpu().numpy()
            actual_list.append(actual_np)

            #for smooth residuals
            # Collect cycles and units for each batch
            cycles_list.append(batch.cycle.cpu().numpy())  # Assuming cycles are stored in batch.cycle
            units_list.append(batch.unit.cpu().numpy())  # Assuming units are stored in batch.unit

    # Concatenate predictions and actuals
    predicted_tensor = np.concatenate(predicted_list, axis=0)
    actual_tensor = np.concatenate(actual_list, axis=0)

    #For smooth residuals
    cycles = np.concatenate(cycles_list, axis=0)
    units = np.concatenate(units_list, axis=0)

    predicted_tensor = predicted_tensor.reshape(-1, num_nodes)  # Shape: [num_samples, num_nodes]
    actual_tensor = actual_tensor.reshape(-1, num_nodes)  # Shape: [num_samples, num_nodes]

    residual = predicted_tensor - actual_tensor

    # Reverse the normalization for both predicted and actual values
    # We no longer have the operational nodes, so just inverse-transform the entire tensors
    predicted_original_scale = sensor_scaler.inverse_transform(predicted_tensor)  # Apply inverse transform to all 14 sensor nodes
    actual_original_scale = sensor_scaler.inverse_transform(actual_tensor)  # Apply inverse transform to all 14 sensor nodes

    residuals_smooth, std_smooth, cycle_smooth, units_smooth = smooth_z_cycle(residual, cycles, units)

    ruls_smooth = list(reversed(cycle_smooth))

    return residual, cycles, units, residuals_smooth, std_smooth, cycle_smooth, units_smooth, predicted_original_scale, actual_original_scale, ruls_smooth


# Step 8: Generate Predictions
test_residuals, test_cycles, test_units, test_residuals_smooth, test_std_smooth, test_cycle_smooth, test_units_smooth, test_predicted_original_scale, test_actual_original_scale, test_ruls_smooth = evaluate(model, test_loader)

# Step 9: Evaluate on Training Units
train_residuals, train_cycles, train_units, train_residuals_smooth, train_std_smooth, train_cycle_smooth, train_units_smooth, train_predicted_original_scale, train_actual_original_scale, train_ruls_smooth = evaluate(model, train_loader_all_cycles)


# Step 10: Plot Predicted vs Actual Sensor Readings
# Sensor names and units (from provided table in the beginning of this script)
sensor_names = [
    "T24", "T30", "T48", "T50",
    "P15", "P2", "P21", "P24", "Ps30", "P40", "P50",
    "Nf", "Nc", "Wf"
]
sensor_units = [
    "°R", "°R", "°R", "°R",
    "psia", "psia", "psia", "psia", "psia", "psia", "psia",
    "rpm", "rpm", "pps"
]

#Save residuals data and other metadata to H5 file (data are all on ALL cycles)
with h5py.File("data_exports/metadata_residuals.h5", "w") as h5f:
    #train
    h5f.create_dataset("residuals_train", data=np.array(train_residuals_smooth))
    h5f.create_dataset("cycles_train", data=np.array(train_cycle_smooth))
    h5f.create_dataset("units_train", data=np.array(train_units_smooth))
    h5f.create_dataset("ruls_train", data=np.array(train_ruls_smooth))
    #test
    h5f.create_dataset("residuals_test", data=np.array(test_residuals_smooth))
    h5f.create_dataset("cycles_test", data=np.array(test_cycle_smooth))
    h5f.create_dataset("units_test", data=np.array(test_units_smooth))
    h5f.create_dataset("ruls_test", data=np.array(test_ruls_smooth))


# Function to plot sensor readings per sensor for each unit
def plot_sensor_readings_per_sensor(actual_data, predicted_data, units, sensor_names, sensor_units, output_dir):
    unique_units = np.unique(units)

    for unit in unique_units:
        # Mask to select data for the current unit
        unit_mask = (units == unit)
        actual_unit = actual_data[unit_mask]
        predicted_unit = predicted_data[unit_mask]

        # Create a figure with 14 subplots arranged in a grid
        fig, axes = plt.subplots(7, 2, figsize=(15, 20))
        fig.suptitle(f'Sensor Readings for Unit {unit}', fontsize=16)

        # Flatten axes array for easy iteration
        axes = axes.flatten()

        for i, sensor in enumerate(sensor_names):
            ax = axes[i]
            actual_sensor_data = actual_unit[:, i]
            predicted_sensor_data = predicted_unit[:, i]
            ax.plot(actual_sensor_data, color="blue", label="Actual")  # Actual readings in blue
            ax.plot(predicted_sensor_data, color="orange", label="Predicted")  # Predicted readings in orange
            ax.set_title(f"{sensor} ({sensor_units[i]})")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel(sensor_units[i])
            ax.legend()

        # Remove any unused subplots (if fewer than 14 sensors are plotted)
        for j in range(len(sensor_names), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
        plt.savefig(f"{output_dir}/unit_{unit}_sensor_readings_per_sensor.png")
        plt.close()


# Function to plot residuals per sensor for each unit
def plot_smoothed_residuals_per_sensor(residuals_smooth, cycles_smooth, units_smooth, sensor_names, sensor_units,
                                       output_dir):
    unique_units = np.unique(units_smooth)

    for unit in unique_units:
        # Mask to select smoothed residuals for the current unit
        unit_mask = (units_smooth == unit)
        residuals_unit_smooth = residuals_smooth[unit_mask]
        cycles_unit_smooth = cycles_smooth[unit_mask]

        # Create a figure with 14 subplots arranged in a grid
        fig, axes = plt.subplots(7, 2, figsize=(15, 20))
        fig.suptitle(f'Smoothed Residuals for Unit {unit}', fontsize=16)

        # Flatten axes array for easy iteration
        axes = axes.flatten()

        for i, sensor in enumerate(sensor_names):
            ax = axes[i]
            sensor_residuals_smooth = residuals_unit_smooth[:, i]
            ax.plot(cycles_unit_smooth, sensor_residuals_smooth, color="red", label="Smoothed Residual")
            ax.set_title(f"{sensor} ({sensor_units[i]})")
            ax.set_xlabel("Cycle (Smooth)")
            ax.set_ylabel(sensor_units[i])
            ax.legend()

        # Remove any unused subplots (if fewer than 14 sensors are plotted)
        for j in range(len(sensor_names), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
        plt.savefig(f"{output_dir}/unit_{unit}_smoothed_residuals_per_sensor.png")
        plt.close()


# Calling functions for plotting sensors (predicted vs actual) of all units:
output_dir = "plots/sensor_readings"
plot_sensor_readings_per_sensor(train_actual_original_scale, train_predicted_original_scale, train_units, sensor_names, sensor_units, output_dir)
plot_sensor_readings_per_sensor(test_actual_original_scale, test_predicted_original_scale, test_units, sensor_names, sensor_units, output_dir)

# Calling functions for plotting residuals of all units:
output_dir = "plots/residuals"
plot_smoothed_residuals_per_sensor(train_residuals_smooth, train_cycle_smooth, train_units_smooth, sensor_names, sensor_units, output_dir)
plot_smoothed_residuals_per_sensor(test_residuals_smooth, test_cycle_smooth, test_units_smooth, sensor_names, sensor_units, output_dir)


