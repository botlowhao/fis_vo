#!/usr/bin/env python3

import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

def data_preprocessing(file_path):
    data = pd.read_csv(file_path)
    Nu = data.iloc[:, 0]
    Er = data.iloc[:, 1]

    Nu_stats = {
        "min": np.min(Nu),
        "max": np.max(Nu),
    }

    Er_stats = {
        "min": np.min(Er),
        "max": np.max(Er),
    }

    return Nu_stats, Er_stats

def create_fis(Nu_stats, Er_stats):
    # Set data for Am1-Am8
    Bm1 = Nu_stats["min"]
    Bm2 = Nu_stats["max"]
    Bm3 = Nu_stats["min"]
    Bm4 = Nu_stats["max"]
    Bm5 = Er_stats["min"]
    Bm6 = Er_stats["max"]
    Bm7 = Er_stats["min"]
    Bm8 = Er_stats["max"]

    # Add Input fuzzy variable p, w
    x_Nu = np.arange(600, 1201, 1)
    x_Er = np.arange(-5.000000, 15.000001, 0.000001)
    x_V1 = np.arange(0, 1.0001, 0.0001)

    # Define fuzzy control variable
    Nu = ctrl.Antecedent(x_Nu, 'Nu')
    Er = ctrl.Antecedent(x_Er, 'Er')
    V1 = ctrl.Consequent(x_V1, 'V1')
    
    # Define input fuzzy membership function
    Nu['B1'] = fuzz.zmf(x_Nu, Bm1, Bm2)
    Nu['B2'] = fuzz.smf(x_Nu, Bm3, Bm4)
    
    Er['B3'] = fuzz.zmf(x_Er, Bm5, Bm6)
    Er['B4'] = fuzz.smf(x_Er, Bm7, Bm8)


    # Define output fuzzy membership function
    V1['Low Variance'] = fuzz.trimf(x_V1, [0, 0.25, 0.50])
    V1['Medium Variance'] = fuzz.trimf(x_V1, [0.25, 0.50, 0.75])
    V1['High Variance'] = fuzz.trimf(x_V1, [0.50, 0.75, 1.00])
    

    # Set defuzzification method 
    V1.defuzzify_method = 'centroid'

    # Display input/output variable and its membership function
    Nu.view()
    Er.view()
    V1.view()

    # Set fuzzy rules
    rule1 = ctrl.Rule(antecedent=(Nu['B2'] & Er['B3']), consequent=V1['Low Variance'], label='R1')
    rule2 = ctrl.Rule(antecedent=(Nu['B2'] & Er['B4']), consequent=V1['High Variance'], label='R2')
    rule3 = ctrl.Rule(antecedent=(Nu['B1'] & Er['B3']), consequent=V1['Medium Variance'], label='R3')
    rule4 = ctrl.Rule(antecedent=(Nu['B1'] & Er['B4']), consequent=V1['High Variance'], label='R4')

    V1_Calculate_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    V1_Calculate = ctrl.ControlSystemSimulation(V1_Calculate_ctrl)

    return V1_Calculate, V1

def test_fis(V1_Calculate):
    # Test for output V1
    V1_Calculate.input['Nu'] = 1008
    V1_Calculate.input['Er'] = 6.001252
    V1_Calculate.compute()
    return V1_Calculate.output['V1']

if __name__ == "__main__":
    file_path = '/home/wyatt/catkin_ws/src/fis_vo/ANFIS_dataset/onlyORBdata/ORB_A.csv'
    Nu_stats, Er_stats = data_preprocessing(file_path)
    V1_Calculate, V1 = create_fis(Nu_stats, Er_stats)
    output_V1 = test_fis(V1_Calculate)
    print(output_V1)
    V1.view(sim=V1_Calculate)
    plt.show()