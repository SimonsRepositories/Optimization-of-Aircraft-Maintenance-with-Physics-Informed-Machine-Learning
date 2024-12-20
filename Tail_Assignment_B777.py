# -*- coding: utf-8 -*-
"""
Created on Sun May  5 20:23:45 2024

@author: BESL
"""

import numpy as np
import datetime as dt
import time
import gurobipy as gp

import out as out
import general as gen
import plot_results as plot_res
from data_prep import prep_data
from nsp.two_sp.mstacbm import MSTACBMProblem


if __name__ == '__main__':   
    inst = {}
    
    # Add data to inst
    inst['logging'] = True
    inst['plotting'] = True
    inst['mask'] = True
    inst['run'] = dt.datetime.now().strftime("%Y%m%d_%H%M")
    inst['minutes_discretization'] = 15
    inst['n_days_defer'] = 0
    inst['cost'] = {"W_due": 10**4, # ^8
                    "W_ground": 10**2, # ^5
                    "mu": 0.6,
                    "delta": 3,
                    "W_defer": 0.1,
                    "W_clean": 10,
                    "W_interval": 10**3,
                    "minutes_slot_fix": 1000, # Fix time (= Cost) needed for each slot
                    "max_free_unused_hours": 18, # Max hours before to early scheduling of task will be penalized
                    'W_slack': 0.5,
                    "C_cancel": 10**10}
    # Problem AC: HBJNF, HBJNG, HBJNH, HBJNK 
    # Base Maintenance: "HBJNF",
    # old dates: 07.04 to 12.04 2024
    inst['aircraft_registration_selection'] = ['HBJNA', 'HBJNB', 'HBJNC','HBJND', 'HBJNE', 'HBJNI', 'HBJNJ', 'HBJNL'
                                               ]
    inst['subtype_selection'] = ['777']
    inst['n_aircraft'] = len(inst['aircraft_registration_selection'])
    inst['start_date'] = dt.datetime.strptime("2024-04-07 00:00:00",
                                             '%Y-%m-%d %H:%M:%S')
    inst['end_date'] = dt.datetime.strptime("2024-04-12 00:00:00", 
                                           '%Y-%m-%d %H:%M:%S')
    inst['n_days'] = (inst['end_date'] - inst['start_date']).days
    inst['hub_airports'] = ['ZRH']
    inst['time'] = {}
    inst['time']['current_run_start_time'] = time.time()
    inst['time']['current_run_datetime'] = dt.datetime.now()

    out.print_and_log(inst['time']['current_run_datetime'])
    
    rng = np.random.RandomState()
    rng.seed(42)
    
    if inst['logging']:
        logger = gen.setup_logging('output/example_{}ac_{}days_{}.txt'.format(
            len(inst['aircraft_registration_selection']),
            (inst['end_date'] - inst['start_date']).days,
            inst['run']))
        
    data, flight_params, mx_params, pg_params = prep_data(
        settings=inst,
        adaption_orders_factor=0, 
        adaption_flights_factor=0,
        adaption_orders_duration_factor=0,
        adaption_orders_due_factor=1,
        ) 
    
    inst['data'] = data    
    inst['flight_params'] = flight_params
    inst['mx_params'] = mx_params
    inst['pg_params'] = pg_params
    inst['time']['current_run_finish_data_setup_time'] = time.time()

    
    reg_date = [inst['aircraft_registration_selection'], [inst['start_date'], inst['end_date']]]
    mstacbm = MSTACBMProblem(inst, reg_date, rng)
    
        
    inst['time']['time_start_extensive'] = time.time()
    
    """
    # Add Scenario
    scenario = mstacbm.get_scenarios(n_scenarios=1, test_set='0')[0]
      
    mstacbm._adaption_flight_schedule(factor=scenario[0])
    mstacbm._adaption_pg_task(sd_adaption=scenario[1:])
    """
    
    model_ex = mstacbm._make_extensive_model()
    model_ex.optimize()
    
    inst['time']['time_model_finish_extensive'] = time.time()

    if model_ex.Status != gp.GRB.INFEASIBLE and model_ex.Status != gp.GRB.LOADED:
        
        file_add = "extensive_{}".format(inst['run'])
        
        # Get flight stats
        res = out.get_flight_stats(inst, mstacbm, inst['flight_params'].flight_schedule)
        df_flight_flown_extensive, df_flight_canceled_extensive, df_flight_counter_extensive = res
        
        # Get mx task stast
        res = out.get_mx_stats_task(inst, mstacbm, inst['mx_params'])
        df_slot_task_scheduled_extensive, df_task_scheduled_extensive, df_task_not_scheduled_extensive, df_slot_used_extensive = res
        
        # Get model variable with selection
        df_dict_res = out.get_model_stats(inst, mstacbm, model=model_ex)  

        if inst['mask']:            
            df_flight_flown_extensive, df_flight_canceled_extensive, df_slot_used_extensive = plot_res.mask(df_flight_flown_extensive, 
                                                                                             df_flight_canceled_extensive, 
                                                                                             df_slot_used_extensive, 
                                                                                             inst) 

        out.print_res(model_ex, 
                      inst, 
                      df_flight_flown_extensive, 
                      df_flight_canceled_extensive, 
                      df_slot_task_scheduled_extensive, 
                      df_task_scheduled_extensive, 
                      df_task_not_scheduled_extensive,
                      pg_params.predictor_names)
 
        if inst['plotting']:
            plot_res.plot_aircraft_schedules_plotly(inst, 
                                                    df_slot_used_extensive, 
                                                    df_flight_flown_extensive, 
                                                    file_add)
    
            plot_res.plot_flightplan_plotly(inst, 
                                            df_flight_flown_extensive, 
                                            df_flight_canceled_extensive, 
                                            df_slot_used_extensive, 
                                            file_add, 
                                            colormap='Dark2')
            
        
    inst['time']['time_end_extensive'] = time.time()
    
 
    
    if inst['logging']:
        gen.close_logging(logger)