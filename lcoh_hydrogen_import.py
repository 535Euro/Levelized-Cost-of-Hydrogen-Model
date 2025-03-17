"""
Sensitivity Analysis Module for Hydrogen Production and Transport

This module provides functions to compute the Levelized Cost of Hydrogen (LCOH) for production,
perform production and transport sensitivity analyses, generate related charts, and combine results
for overall sensitivity analysis.

Expected Excel Input File Structure:
    The Excel file must contain the following sheets with the specified structure:
    
    1. Sheet "production":
         - Contains production input parameters.
         - Expected columns include:
             * "parameter": The name of the parameter (e.g., CAPEX, WACC, n, OPEX, FLH, E, eta_sys, stack_lifetime).
             * "baseline": The baseline value for each parameter.
             * "min": The minimum value to be used in the sensitivity analysis.
             * "max": The maximum value to be used in the sensitivity analysis.
             * "step": The incremental step for varying the parameter.
    
    2. Sheet "transport":
         - Contains transport cost parameters.
         - Expected columns include:
             * "parameter": The name of the transport parameter (e.g., LH2_Ship_rate, Ammonia_Ship_rate, LOHC_Ship_rate, pipeline_cost_rate_onshore).
             * "baseline": The baseline cost rate value.
             * "min": The minimum value for the sensitivity analysis.
             * "max": The maximum value for the sensitivity analysis.
             * "step": The incremental step for varying the cost rate.
    
    3. Sheet "routes":
         - Contains route data for transport analysis.
         - Expected columns include:
             * "from": Departure port.
             * "to": Destination port.
             * "baseline_distance": The shipping distance used as baseline (in km).
             * "cluster": The import cluster.
             * "compatible_derivatives": A comma-separated list of derivatives compatible with the route.
    
    4. Sheet "pipeline_europe":
         - Contains European pipeline data.
         - Expected columns include:
             * "ports": The port names.
             * "distance_german_border": The distance to the German border (in km), used to estimate pipeline costs.

Authors:
    - Justus Hünicke
    - Joscha Waldau
    - Carlos Pohl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------------
# Function: compute_production_lcoh
#
# Calculates the Levelized Cost of Hydrogen (LCOH) for production using the formula:
#
#   LCOH = (LHV / η_sys) * [ ( (WACC/100 * (1+WACC/100)^n) / ((1+WACC/100)^n - 1) + OPEX/100 )
#                           * (CAPEX / FLH) + E ]
#
# Parameters:
#   CAPEX   - Capital Expenditure (€/kW)
#   WACC    - Discount rate (%)
#   n       - Lifetime (years)
#   OPEX    - Operational expenditure (% of CAPEX)
#   FLH     - Full load hours per year
#   E       - Electricity cost (€/kWh)
#   eta_sys - System efficiency (decimal)
#   LHV     - Lower heating value (default: 33.3 kWh/kg)
#
# Returns:
#   LCOH value (€/kg H2)
# --------------------------------------------------------------------------------
def compute_production_lcoh(CAPEX, WACC, n, OPEX, FLH, E, eta_sys, LHV=33.3):
    i = WACC
    annuity_factor = ((i/100 * (1 + i/100)**n) / ((1 + i/100)**n - 1)) if ((1 + i/100)**n - 1) != 0 else np.nan
    cost_component = annuity_factor + (OPEX/100)
    LCOH = (LHV / eta_sys) * ((CAPEX / FLH) * cost_component + E)
    return LCOH

# --------------------------------------------------------------------------------
# Function: compute_transport_cost_components
#
# Computes shipping and pipeline costs using a single pipeline cost parameter.
#
#   - Shipping cost = (distance / 1000) * shipping_rate
#   - Pipeline cost = (pipeline_distance / 1000) * pipeline_rate_onshore
#
# The overall cost is the sum of shipping and pipeline costs.
#
# Returns:
#   (shipping_cost, pipeline_cost, overall_cost)
# --------------------------------------------------------------------------------
def compute_transport_cost_components(distance, shipping_rate, pipeline_distance=0, pipeline_rate_onshore=None):
    shipping_cost = (distance / 1000) * shipping_rate
    pipeline_cost = 0
    if pipeline_distance > 0 and pipeline_rate_onshore is not None:
        pipeline_cost = (pipeline_distance / 1000) * pipeline_rate_onshore
    overall_cost = shipping_cost + pipeline_cost
    return shipping_cost, pipeline_cost, overall_cost

# --------------------------------------------------------------------------------
# Function: production_sensitivity_analysis
#
# Performs a one-at-a-time sensitivity analysis on production parameters (excluding "stack_lifetime").
# Also computes worst-case and best-case production scenarios.
#
# Returns:
#   (prod_sens_df, worst_scenario, best_scenario, baseline_values)
# --------------------------------------------------------------------------------
def production_sensitivity_analysis(prod_df):
    results = []
    baseline_values = {row['parameter'].strip(): row['baseline'] for idx, row in prod_df.iterrows()}
    
    for idx, row in prod_df.iterrows():
        param = row['parameter'].strip()
        if param.lower() == 'stack_lifetime':
            continue
        values = np.arange(row['min'], row['max'] + row['step']/2, row['step'])
        for val in values:
            params = baseline_values.copy()
            params[param] = val
            lcoh = compute_production_lcoh(params['CAPEX'], params['WACC'], params['n'],
                                           params['OPEX'], params['FLH'], params['E'], params['eta_sys'])
            deviation = ((val - row['baseline']) / row['baseline'])*100 if row['baseline'] != 0 else 0
            results.append({
                'Varied Parameter': param,
                'Parameter Value': val,
                'Deviation (%)': deviation,
                'CAPEX': params['CAPEX'],
                'WACC': params['WACC'],
                'n': params['n'],
                'OPEX': params['OPEX'],
                'FLH': params['FLH'],
                'E': params['E'],
                'eta_sys': params['eta_sys'],
                'LCOH': lcoh
            })
    prod_sens_df = pd.DataFrame(results)
    
    worst_params = {
        'CAPEX': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='capex', 'max'].values[0],
        'WACC': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='wacc', 'max'].values[0],
        'n': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='n', 'min'].values[0],
        'OPEX': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='opex', 'max'].values[0],
        'FLH': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='flh', 'min'].values[0],
        'E': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='e', 'max'].values[0],
        'eta_sys': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='eta_sys', 'min'].values[0]
    }
    worst_LCOH = compute_production_lcoh(worst_params['CAPEX'], worst_params['WACC'], worst_params['n'],
                                         worst_params['OPEX'], worst_params['FLH'], worst_params['E'], worst_params['eta_sys'])
    worst_scenario = {'Scenario': 'Worst-case Production', **worst_params, 'LCOH': worst_LCOH}
    
    best_params = {
        'CAPEX': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='capex', 'min'].values[0],
        'WACC': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='wacc', 'min'].values[0],
        'n': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='n', 'max'].values[0],
        'OPEX': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='opex', 'min'].values[0],
        'FLH': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='flh', 'max'].values[0],
        'E': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='e', 'min'].values[0],
        'eta_sys': prod_df.loc[prod_df['parameter'].str.lower().str.strip()=='eta_sys', 'max'].values[0]
    }
    best_LCOH = compute_production_lcoh(best_params['CAPEX'], best_params['WACC'], best_params['n'],
                                        best_params['OPEX'], best_params['FLH'], best_params['E'], best_params['eta_sys'])
    best_scenario = {'Scenario': 'Best-case Production', **best_params, 'LCOH': best_LCOH}
    
    return prod_sens_df, worst_scenario, best_scenario, baseline_values

# --------------------------------------------------------------------------------
# Function: production_sensitivity_chart
#
# Plots the production sensitivity chart (LCOH vs. Deviation from baseline) for each
# production parameter. Saves as "production_sensitivity_chart.png".
# --------------------------------------------------------------------------------
def production_sensitivity_chart(prod_sens_df):
    plt.figure(figsize=(10,6))
    parameters = prod_sens_df['Varied Parameter'].unique()
    for param in parameters:
        df_param = prod_sens_df[prod_sens_df['Varied Parameter'] == param]
        plt.plot(df_param['Deviation (%)'], df_param['LCOH'], marker='o', label=param)
    plt.xlabel('Deviation from Baseline (%)')
    plt.ylabel('Production Costs (€/kg H2)')
    plt.title('Production Sensitivity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("production_sensitivity_chart.png")
    plt.show()

# --------------------------------------------------------------------------------
# Function: aggregate_routes_by_export_and_cluster
#
# Aggregates route data by export hub and import cluster. For each combination, calculates:
#   - Average shipping distance (avg_distance)
#   - Average pipeline distance (avg_pipeline_distance)
#   - Combined unique derivatives (compatible_derivatives)
#
# Returns a DataFrame with columns: Export Hub, cluster, avg_distance, compatible_derivatives, avg_pipeline_distance
# --------------------------------------------------------------------------------
def aggregate_routes_by_export_and_cluster(routes_df, pipeline_europe_df):
    # Rename 'from' to 'Export Hub' to avoid conflicts.
    routes_df = routes_df.rename(columns={'from': 'Export Hub'})
    
    # Group by Export Hub and cluster.
    aggregated = routes_df.groupby(['Export Hub', 'cluster']).agg({'baseline_distance': 'mean'}).reset_index()
    aggregated = aggregated.rename(columns={'baseline_distance': 'avg_distance'})
    
    # Combine unique derivatives per group.
    agg_deriv = routes_df.groupby(['Export Hub', 'cluster'])['compatible_derivatives'].apply(
        lambda x: ','.join(sorted(set([d.strip() for item in x.dropna() for d in item.split(',') if d.strip() != ''])))
    ).reset_index()
    aggregated = aggregated.merge(agg_deriv, on=['Export Hub', 'cluster'])
    
    # Standardize 'to' and merge with pipeline data.
    routes_df['to_clean'] = routes_df['to'].str.lower().str.strip()
    pipeline_europe_df['ports_clean'] = pipeline_europe_df['ports'].str.lower().str.strip()
    merged = pd.merge(routes_df, pipeline_europe_df, left_on='to_clean', right_on='ports_clean', how='left')
    agg_pipeline = merged.groupby(['Export Hub', 'cluster'])['distance_german_border'].mean().reset_index()
    aggregated = aggregated.merge(agg_pipeline, on=['Export Hub', 'cluster'], how='left')
    aggregated['distance_german_border'] = aggregated['distance_german_border'].fillna(0)
    aggregated = aggregated.rename(columns={'distance_german_border': 'avg_pipeline_distance'})
    return aggregated

# --------------------------------------------------------------------------------
# Function: transport_sensitivity_analysis
#
# Performs a sensitivity analysis on transport costs for each export hub and import cluster.
# For each export hub–cluster pair and for each derivative in that cluster,
# the shipping parameter (or the pipeline cost parameter) is varied.
#
# Returns a DataFrame with columns:
#   Cluster, Export Hub, Derivative, Varied Parameter, Parameter Value, Deviation (%),
#   Distance (km), Pipeline Distance (km),
#   Shipping Rate (€/kg per 1000km), Pipeline Rate (€/kg per 1000km),
#   Shipping Cost (€/kg), Pipeline Cost (€/kg), Transport Cost (€/kg)
# --------------------------------------------------------------------------------
def transport_sensitivity_analysis(routes_df, transport_df, pipeline_europe_df):
    aggregated = aggregate_routes_by_export_and_cluster(routes_df, pipeline_europe_df)
    results = []
    
    # Identify the pipeline cost parameter row.
    pipeline_row = transport_df[transport_df['parameter'].str.strip() == 'pipeline_cost_rate_onshore'].iloc[0]
    pipeline_baseline = pipeline_row['baseline']
    pipeline_min = pipeline_row['min']
    pipeline_max = pipeline_row['max']
    pipeline_step = pipeline_row['step']
    
    for idx, row in aggregated.iterrows():
        export_hub = row['Export Hub']
        cluster = row['cluster']
        avg_distance = row['avg_distance']
        avg_pipeline_distance = row['avg_pipeline_distance']
        
        # Get derivatives as list.
        deriv_list = [d.strip() for d in row['compatible_derivatives'].split(',') if d.strip() != '']
        
        for deriv in deriv_list:
            deriv_lower = deriv.lower()
            if 'lh2' in deriv_lower:
                ship_param = 'LH2_Ship_rate'
            elif 'ammonia' in deriv_lower:
                ship_param = 'Ammonia_Ship_rate'
            elif 'lohc' in deriv_lower:
                ship_param = 'LOHC_Ship_rate'
            else:
                continue
            
            # Retrieve shipping parameter details.
            ship_row = transport_df[transport_df['parameter'].str.strip() == ship_param].iloc[0]
            baseline_ship = ship_row['baseline']
            ship_min = ship_row['min']
            ship_max = ship_row['max']
            ship_step = ship_row['step']
            
            # 1) Vary shipping parameter while keeping pipeline cost fixed at baseline.
            ship_values = np.arange(ship_min, ship_max + ship_step/2, ship_step)
            for val in ship_values:
                deviation = ((val - baseline_ship) / baseline_ship) * 100 if baseline_ship != 0 else 0
                shipping_rate = val
                pipeline_rate = pipeline_baseline
                ship_cost, pipe_cost, overall_cost = compute_transport_cost_components(
                    avg_distance, shipping_rate, avg_pipeline_distance, pipeline_rate
                )
                results.append({
                    'Cluster': cluster,
                    'Export Hub': export_hub,
                    'Derivative': deriv,
                    'Varied Parameter': ship_param,
                    'Parameter Value': val,
                    'Deviation (%)': deviation,
                    'Distance (km)': avg_distance,
                    'Pipeline Distance (km)': avg_pipeline_distance,
                    'Shipping Rate (€/kg per 1000km)': shipping_rate,
                    'Pipeline Rate (€/kg per 1000km)': pipeline_rate,
                    'Shipping Cost (€/kg)': ship_cost,
                    'Pipeline Cost (€/kg)': pipe_cost,
                    'Transport Cost (€/kg)': overall_cost
                })
            
            # 2) Vary pipeline parameter while keeping shipping rate fixed at baseline.
            pipeline_values = np.arange(pipeline_min, pipeline_max + pipeline_step/2, pipeline_step)
            for val in pipeline_values:
                deviation = ((val - pipeline_baseline) / pipeline_baseline) * 100 if pipeline_baseline != 0 else 0
                shipping_rate = baseline_ship
                pipeline_rate = val
                ship_cost, pipe_cost, overall_cost = compute_transport_cost_components(
                    avg_distance, shipping_rate, avg_pipeline_distance, pipeline_rate
                )
                results.append({
                    'Cluster': cluster,
                    'Export Hub': export_hub,
                    'Derivative': deriv,
                    'Varied Parameter': 'pipeline_cost_rate_onshore',
                    'Parameter Value': val,
                    'Deviation (%)': deviation,
                    'Distance (km)': avg_distance,
                    'Pipeline Distance (km)': avg_pipeline_distance,
                    'Shipping Rate (€/kg per 1000km)': shipping_rate,
                    'Pipeline Rate (€/kg per 1000km)': pipeline_rate,
                    'Shipping Cost (€/kg)': ship_cost,
                    'Pipeline Cost (€/kg)': pipe_cost,
                    'Transport Cost (€/kg)': overall_cost
                })
    transport_sens_df = pd.DataFrame(results)
    return transport_sens_df

# --------------------------------------------------------------------------------
# Function: transport_sensitivity_chart_ui
#
# Provides an interactive UI to generate a transport sensitivity chart.
# The user is prompted for a departure port, a destination cluster, and transport types.
# For each selected mode:
#   - For shipping modes (LH2, Ammonia, LOHC): pure shipping cost vs. deviation.
#   - For pipeline: pure pipeline cost vs. deviation.
#
# The resulting plot is saved and displayed.
# --------------------------------------------------------------------------------
def transport_sensitivity_chart_ui(routes_df, transport_df, pipeline_europe_df):
    import matplotlib.pyplot as plt

    # Display available options.
    possible_from_ports = sorted(routes_df['from'].unique())
    possible_clusters = sorted(routes_df['cluster'].unique())
    possible_transport_types = ["LH2", "Ammonia", "LOHC", "Pipeline"]

    print("Available departure ports:", ", ".join(possible_from_ports))
    print("Available destination clusters:", ", ".join(possible_clusters))
    print("Available transport types:", ", ".join(possible_transport_types))

    dep_port = input("Enter departure port exactly as shown above: ").strip()
    dest_cluster = input("Enter destination cluster exactly as shown above: ").strip()
    transport_types_input = input("Enter transport types (comma separated, e.g. LH2, Ammonia, LOHC, Pipeline): ").strip()
    selected_types = [tt.strip().lower() for tt in transport_types_input.split(',')]

    # Filter routes for given export hub and cluster.
    filtered_routes = routes_df[
        (routes_df['from'].str.lower() == dep_port.lower()) &
        (routes_df['cluster'].str.lower() == dest_cluster.lower())
    ].copy()
    if filtered_routes.empty:
        print("No routes found for the given departure port and destination cluster.")
        return

    # Calculate average shipping distance for this export hub and cluster.
    avg_distance = filtered_routes['baseline_distance'].mean()

    # Merge with pipeline_europe_df to obtain average pipeline distance.
    pipeline_europe_df['ports_clean'] = pipeline_europe_df['ports'].str.lower().str.strip()
    filtered_routes['to_clean'] = filtered_routes['to'].str.lower().str.strip()
    merged = pd.merge(filtered_routes, pipeline_europe_df, left_on='to_clean', right_on='ports_clean', how='left')
    avg_pipeline_distance = merged['distance_german_border'].mean()
    if pd.isna(avg_pipeline_distance):
        avg_pipeline_distance = 0

    derivative_lines = {}
    all_deviations = []

    # For shipping modes.
    for t in selected_types:
        if t not in ["lh2", "ammonia", "lohc"]:
            continue
        if t == "lh2":
            shipping_param = "LH2_Ship_rate"
        elif t == "ammonia":
            shipping_param = "Ammonia_Ship_rate"
        elif t == "lohc":
            shipping_param = "LOHC_Ship_rate"
        else:
            print(f"Transport type '{t}' not recognized. Skipping.")
            continue
        
        row = transport_df[transport_df['parameter'].str.strip() == shipping_param].iloc[0]
        baseline_val = row['baseline']
        min_val = row['min']
        max_val = row['max']
        step_val = row['step']

        x_vals = []
        y_vals = []
        param_values = np.arange(min_val, max_val + step_val/2, step_val)
        for val in param_values:
            deviation = ((val - baseline_val)/baseline_val)*100 if baseline_val != 0 else 0
            x_vals.append(deviation)
            shipping_cost = (avg_distance/1000.0) * val
            y_vals.append(shipping_cost)
        derivative_lines[t.upper()] = (np.array(x_vals), np.array(y_vals))
        all_deviations.extend(x_vals)

    # For Pipeline.
    if "pipeline" in selected_types:
        pipeline_row = transport_df[transport_df['parameter'].str.strip() == 'pipeline_cost_rate_onshore'].iloc[0]
        pipeline_baseline = pipeline_row['baseline']
        pipeline_min = pipeline_row['min']
        pipeline_max = pipeline_row['max']
        pipeline_step = pipeline_row['step']
        
        param_values = np.arange(pipeline_min, pipeline_max + pipeline_step/2, pipeline_step)
        x_vals_p = []
        y_vals_p = []
        for val in param_values:
            deviation = ((val - pipeline_baseline)/pipeline_baseline)*100 if pipeline_baseline != 0 else 0
            x_vals_p.append(deviation)
            pipeline_cost = (avg_pipeline_distance/1000.0) * val
            y_vals_p.append(pipeline_cost)
        derivative_lines["PIPELINE"] = (np.array(x_vals_p), np.array(y_vals_p))
        all_deviations.extend(x_vals_p)

    if all_deviations:
        x_min = min(all_deviations)
        x_max = max(all_deviations)
    else:
        x_min, x_max = -50, 50

    x_axis = np.linspace(x_min, x_max, 100)

    plt.figure(figsize=(10,6))
    for label_str, (x_vals, y_vals) in derivative_lines.items():
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=label_str)
    plt.xlabel('Deviation from Baseline Parameter (%)')
    plt.ylabel('Pure Transport Cost (€/kg H2)')
    title_str = f"Transport Sensitivity: {dep_port} to {dest_cluster}"
    plt.title(title_str)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"transport_sensitivity_{dep_port.replace(' ', '_')}_{dest_cluster.replace(' ', '_')}_single_pipeline.png"
    plt.savefig(filename)
    plt.show()
    print(f"Transport sensitivity chart saved as '{filename}'.")

# --------------------------------------------------------------------------------
# Function: overall_sensitivity_analysis
#
# Combines production LCOH with transport costs (baseline, worst, best) to compute overall LCOH.
# For each group (by Cluster, Export Hub, and Derivative), it adds production cost (baseline, worst, best)
# to the corresponding transport cost to yield overall values.
# --------------------------------------------------------------------------------
def overall_sensitivity_analysis(prod_baseline, worst_prod, best_prod, transport_sens_df):
    overall_results = []
    baseline_prod_LCOH = compute_production_lcoh(prod_baseline['CAPEX'], prod_baseline['WACC'], prod_baseline['n'],
                                                 prod_baseline['OPEX'], prod_baseline['FLH'], prod_baseline['E'],
                                                 prod_baseline['eta_sys'])
    worst_prod_LCOH = worst_prod['LCOH']
    best_prod_LCOH = best_prod['LCOH']
    
    group_cols = ['Cluster', 'Export Hub', 'Derivative']
    grouped = transport_sens_df.groupby(group_cols)
    for group_keys, group in grouped:
        baseline_rows = group[np.isclose(group['Deviation (%)'], 0)]
        if not baseline_rows.empty:
            baseline_transport = baseline_rows['Transport Cost (€/kg)'].median()
        else:
            baseline_transport = group['Transport Cost (€/kg)'].median()
        worst_transport = group['Transport Cost (€/kg)'].max()
        best_transport = group['Transport Cost (€/kg)'].min()
        overall_baseline = baseline_prod_LCOH + baseline_transport
        overall_worst = worst_prod_LCOH + worst_transport
        overall_best = best_prod_LCOH + best_transport
        overall_results.append({
            'Cluster': group_keys[0],
            'Export Hub': group_keys[1],
            'Derivative': group_keys[2],
            'Production LCOH Baseline (€/kg)': baseline_prod_LCOH,
            'Production LCOH Worst (€/kg)': worst_prod_LCOH,
            'Production LCOH Best (€/kg)': best_prod_LCOH,
            'Transport Cost Baseline (€/kg)': baseline_transport,
            'Transport Cost Worst (€/kg)': worst_transport,
            'Transport Cost Best (€/kg)': best_transport,
            'Overall LCOH Baseline (€/kg)': overall_baseline,
            'Overall LCOH Worst (€/kg)': overall_worst,
            'Overall LCOH Best (€/kg)': overall_best
        })
    overall_df = pd.DataFrame(overall_results)
    return overall_df

# --------------------------------------------------------------------------------
# Main function
#
# 1) Reads input Excel (production, transport, routes, pipeline_europe).
# 2) Performs production sensitivity analysis and plots the production sensitivity chart.
# 3) Appends worst- and best-case production scenarios to the production results.
# 4) Performs transport sensitivity analysis for each export hub individually.
# 5) Performs overall sensitivity analysis by combining production and transport results.
# 6) Writes outputs to an Excel file.
# 7) Repeatedly prompts the user to generate transport sensitivity charts until 'n' is entered.
# --------------------------------------------------------------------------------
def main():
    excel_path = input("Please enter the path to the Excel file with input parameters: ").strip()
    if not os.path.exists(excel_path):
        print("File not found.")
        return

    # Read input sheets.
    prod_df = pd.read_excel(excel_path, sheet_name='production')
    transport_df = pd.read_excel(excel_path, sheet_name='transport')
    routes_df = pd.read_excel(excel_path, sheet_name='routes')
    pipeline_europe_df = pd.read_excel(excel_path, sheet_name='pipeline_europe')

    # Clean parameter columns.
    prod_df['parameter'] = prod_df['parameter'].str.strip()
    transport_df['parameter'] = transport_df['parameter'].str.strip()

    # Production sensitivity analysis.
    prod_sens_df, worst_prod, best_prod, prod_baseline = production_sensitivity_analysis(prod_df)
    production_sensitivity_chart(prod_sens_df)
    
    # Append worst-case and best-case production scenarios.
    worst_row = {
        "Varied Parameter": worst_prod.get("Scenario", "Worst-case Production"),
        "Parameter Value": np.nan,
        "Deviation (%)": np.nan,
        "CAPEX": worst_prod.get("CAPEX"),
        "WACC": worst_prod.get("WACC"),
        "n": worst_prod.get("n"),
        "OPEX": worst_prod.get("OPEX"),
        "FLH": worst_prod.get("FLH"),
        "E": worst_prod.get("E"),
        "eta_sys": worst_prod.get("eta_sys"),
        "LCOH": worst_prod.get("LCOH")
    }
    best_row = {
        "Varied Parameter": best_prod.get("Scenario", "Best-case Production"),
        "Parameter Value": np.nan,
        "Deviation (%)": np.nan,
        "CAPEX": best_prod.get("CAPEX"),
        "WACC": best_prod.get("WACC"),
        "n": best_prod.get("n"),
        "OPEX": best_prod.get("OPEX"),
        "FLH": best_prod.get("FLH"),
        "E": best_prod.get("E"),
        "eta_sys": best_prod.get("eta_sys"),
        "LCOH": best_prod.get("LCOH")
    }
    worst_df = pd.DataFrame([worst_row])
    best_df = pd.DataFrame([best_row])
    prod_sens_df_final = pd.concat([prod_sens_df, worst_df, best_df], ignore_index=True)

    # Transport sensitivity analysis for each export hub and cluster.
    transport_sens_df = transport_sensitivity_analysis(routes_df, transport_df, pipeline_europe_df)
    
    # Overall sensitivity analysis.
    overall_df = overall_sensitivity_analysis(prod_baseline, worst_prod, best_prod, transport_sens_df)

    # Write output Excel file.
    output_file = "sensitivity_analysis_output.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        prod_sens_df_final.to_excel(writer, sheet_name='Production Sensitivity', index=False)
        transport_sens_df.to_excel(writer, sheet_name='Transport Sensitivity', index=False)
        overall_df.to_excel(writer, sheet_name='Overall Sensitivity', index=False)
    print(f"Excel output saved as '{output_file}'.")

    # Repeatedly ask if the user wants to generate a specific transport sensitivity chart.
    while True:
        answer = input("Do you want to generate a specific transport sensitivity chart? (y/n): ").strip().lower()
        if answer == 'y':
            transport_sensitivity_chart_ui(routes_df, transport_df, pipeline_europe_df)
        else:
            break

    print("Analysis complete.")

if __name__ == "__main__":
    main()
