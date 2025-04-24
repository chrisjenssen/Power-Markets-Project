
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd


from funskjoner import read_excel_file, create_y_matrix, generate_Y_DC

filename = 'Problem_2_data.xlsx'
sheet_1 = 'Problem 2.2 - Base case'
sheet_2 = 'Problem 2.3 - Generators'
sheet_3 = 'Problem 2.4 - Loads'
sheet_4 = 'Problem 2.5 - Environmental'

generator, load, transmission = read_excel_file(filename, sheet_1)
num_buses = 3  # Set the number of buses in your system, have to do this manually due to the set up in excel file
Y_bus = create_y_matrix(num_buses, transmission)
Y_DC = generate_Y_DC(generator, Y_bus)

""" ---- Set up the optimization model ---- """
model = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

""" ---- Sets ---- """

model.N = pyo.Set(ordered = True, initialize = load['Location.1'].unique().tolist())   #Set for nodes

model.G = pyo.Set(ordered=True, initialize=generator['Generator'].tolist())

model.T = pyo.Set(ordered = True, initialize = transmission['Line'].tolist())  #Set for transmission lines

""" ---- Parameters ----"""

#Nodes

model.Demand    = pyo.Param(model.N, initialize = load.set_index('Location.1')['Demand [MW]'].to_dict())  #Parameter for demand for every node

#Generators

model.Capacity_gen = pyo.Param(model.G, initialize=generator.set_index('Generator')['Capacity [MW]'].to_dict())

model.MarginalCost = pyo.Param(model.G, initialize=generator.set_index('Generator')['Marginal cost NOK/MWh]'].to_dict())

model.Pu_base   = pyo.Param(initialize = 1000)                  #Parameter for per unit factor

#Transmission lines

model.Capacity_trans  = pyo.Param(model.T, initialize = transmission.set_index('Line')["Capacity [MW].1"].to_dict())

model.Susceptance  = pyo.Param(model.T, initialize = transmission.set_index('Line')["Susceptance [p.u]"].to_dict())      #Parameter for max transfer from node, for every line

model.GenLocation = pyo.Param(
    model.G,
    initialize=generator.set_index('Generator')['Location'].to_dict()
)

# --- Build from/to node mappings for each line ---
# this assumes your transmission DataFrame has a 'Line' column like "Line 1-2"
splits = transmission['Line'].str.extract(r'Line\s*(\d+)-(\d+)')
transmission['from_node'] = 'Node ' + splits[0]
transmission['to_node']   = 'Node ' + splits[1]

# Add those mappings as Pyomo Params
model.transmission_from = pyo.Param(
    model.T,
    initialize=transmission.set_index('Line')['from_node'].to_dict()
)
model.transmission_to   = pyo.Param(
    model.T,
    initialize=transmission.set_index('Line')['to_node'].to_dict()
)

""" ---- Variables ---- """

# Generation levels for each generator
model.gen = pyo.Var(model.G, within=pyo.NonNegativeReals)

# Power flow through each transmission line
model.flow_trans = pyo.Var(model.T)

# Voltage angle at each node
model.theta = pyo.Var(model.N)

#Binary Variables, to model on/off decisions (e.g., whether a generator is running)
model.gen_status = pyo.Var(model.G, within=pyo.Binary)

# Load shedding at each node (if applicable, dont know if it is)
#model.shed = pyo.Var(model.N, within=pyo.NonNegativeReals)

"""
Objective function:
Maximize social welfare
"""

def social_welfare_rule(model):
    # Calculate the total benefit (consumer surplus)
    total_benefit = sum(model.Demand[n] * model.theta[n] for n in model.N)  # Assuming theta represents the price signal

    # Calculate the total cost (producer surplus)
    total_cost = sum(model.gen[g] * model.MarginalCost[g] for g in model.G) # if applicable: + sum(model.shed[n] * model.Cost_shed for n in model.N)

    # Social welfare is the total benefit minus the total cost
    return total_benefit - total_cost

# Define the objective function
model.OBJ = pyo.Objective(rule=social_welfare_rule, sense=pyo.maximize)


"""
Constraints
"""
# Minimum generation constraint
def min_gen_rule(model, g):
    return model.gen[g] >= model.Capacity_gen[g] * model.gen_status[g]

model.min_gen_const = pyo.Constraint(model.G, rule=min_gen_rule)

# Maximum generation constraint
def max_gen_rule(model, g):
    return model.gen[g] <= model.Capacity_gen[g] * model.gen_status[g]

model.max_gen_const = pyo.Constraint(model.G, rule=max_gen_rule)


# Maximum flow constraint for transmission lines
def max_flow_trans_rule(model, t):
    return model.flow_trans[t] <= model.Capacity_trans[t]

model.max_flow_trans_const = pyo.Constraint(model.T, rule=max_flow_trans_rule)

# Minimum flow constraint for transmission lines (if bidirectional flow is allowed)
def min_flow_trans_rule(model, t):
    return model.flow_trans[t] >= -model.Capacity_trans[t]

model.min_flow_trans_const = pyo.Constraint(model.T, rule=min_flow_trans_rule)

# Set the reference node to have a theta == 0
# Extract the reference node
reference_node = generator.loc[generator['Slack bus'], 'Location'].values[0]

# Reference node constraint
def ref_node_rule(model):
    return model.theta[reference_node] == 0

model.ref_node_const = pyo.Constraint(rule=ref_node_rule)

"""
# Power balance constraint
def power_balance_rule(model, n):
    return (sum(model.gen[g] for g in model.G if generator.loc[generator['Generator'] == g, 'Location'].values[0] == n) ==
            model.Demand[n] +
            sum(model.flow_trans[t] for t in model.T if model.Capacity_trans[t] == n) -
            sum(model.flow_trans[t] for t in model.T if model.Capacity_trans[t] == n))

model.power_balance_const = pyo.Constraint(model.N, rule=power_balance_rule)

# Flow balance constraint
def flow_balance_rule(model, t):
    from_node = model.Capacity_trans[t]
    to_node = model.Capacity_trans[t]
    return model.flow_trans[t] == model.Susceptance[t] * (model.theta[from_node] - model.theta[to_node])

model.flow_balance_const = pyo.Constraint(model.T, rule=flow_balance_rule)
"""


"""
# Load balance constraint
def load_balance_rule(model, n):
    return (sum(model.gen[g] for g in model.G if generator.loc[generator['Generator'] == g, 'Location'].values[0] == n) ==
            model.Demand[n] +
            sum(Y_bus[n-1][o-1] * model.theta[o] * model.Pu_base for o in model.N) +
            #sum(Y_DC[h-1][n-1] * model.flow_DC[h] for h in model.H))

model.load_balance_const = pyo.Constraint(model.N, rule=load_balance_rule)
"""

def load_balance_rule(model, n):
    # total generation at node n
    gen_sum = sum(
        model.gen[g]
        for g in model.G
        if model.GenLocation[g] == n
    )

    # AC flows into and out of n
    ac_in  = sum(
        model.flow_trans[t]
        for t in model.T
        if model.transmission_to[t] == n
    )
    ac_out = sum(
        model.flow_trans[t]
        for t in model.T
        if model.transmission_from[t] == n
    )

    # balance: generation + inflow == demand + outflow
    return gen_sum + ac_in == model.Demand[n] + ac_out

model.load_balance_const = pyo.Constraint(model.N, rule=load_balance_rule)


# --- Power balance at each node ---
def power_balance_rule(model, n):
    # total generation at node n
    gen_sum = sum(
        model.gen[g]
        for g in model.G
        if generator.set_index('Generator')['Location'][g] == n
    )
    # total inflow to n
    inflow = sum(
        model.flow_trans[t]
        for t in model.T
        if model.transmission_to[t] == n
    )
    # total outflow from n
    outflow = sum(
        model.flow_trans[t]
        for t in model.T
        if model.transmission_from[t] == n
    )
    # balance: generation + inflow == demand + outflow
    return gen_sum + inflow == model.Demand[n] + outflow

model.power_balance_const = pyo.Constraint(model.N, rule=power_balance_rule)


# --- DC flow law on each line ---
def flow_balance_rule(model, t):
    from_n = model.transmission_from[t]
    to_n   = model.transmission_to[t]
    # B_ij*(θ_i − θ_j)
    return model.flow_trans[t] == model.Susceptance[t] * (
        model.theta[from_n] - model.theta[to_n]
    )

model.flow_balance_const = pyo.Constraint(model.T, rule=flow_balance_rule)


# Print the constraints to verify
print("Minimum Generation Constraint (model.min_gen_const):")
model.min_gen_const.pprint()
print("\nMaximum Generation Constraint (model.max_gen_const):")
model.max_gen_const.pprint()
print("\nMaximum Flow Constraint (model.max_flow_trans_const):")
model.max_flow_trans_const.pprint()
print("\nMinimum Flow Constraint (model.min_flow_trans_const):")
model.min_flow_trans_const.pprint()
print("\nPower Balance Constraint (model.power_balance_const):")
model.power_balance_const.pprint()
print("\nReference Node Constraint (model.ref_node_const):")
model.ref_node_const.pprint()
print("\nFlow Balance Constraint (model.flow_balance_const):")
model.flow_balance_const.pprint()
    
"""
Compute the optimization problem
"""

# Set the solver for this
# opt = SolverFactory("glpk")  # Uncomment if you want to use GLPK
opt = SolverFactory('gurobi', solver_io="python")

# Enable dual variable reading -> important for dual values of results
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# Solve the problem
results = opt.solve(model, load_solutions=True)

# Write result on performance
results.write()

# --- Generator results ---
gen_df = pd.DataFrame({
    'Generator': list(model.G),
    'Online?':    [int(pyo.value(model.gen_status[g])) for g in model.G],
    'Output (MW)': [pyo.value(model.gen[g]) for g in model.G],
    'Capacity (MW)': [model.Capacity_gen[g] for g in model.G],
})
print("\n=== Generator Dispatch ===")
print(gen_df.to_string(index=False))

# --- Line flows ---
flow_df = pd.DataFrame({
    'Line':      list(model.T),
    'From':      [model.transmission_from[t] for t in model.T],
    'To':        [model.transmission_to[t]   for t in model.T],
    'Flow (MW)': [pyo.value(model.flow_trans[t]) for t in model.T],
    'Cap (MW)':  [model.Capacity_trans[t]      for t in model.T],
})
print("\n=== Transmission Flows ===")
print(flow_df.to_string(index=False))

# --- Voltage angles ---
theta_df = pd.DataFrame({
    'Node':      list(model.N),
    'Angle (rad)': [pyo.value(model.theta[n]) for n in model.N],
    'Demand (MW)': [model.Demand[n] for n in model.N],
})
print("\n=== Nodal Angles & Demand ===")
print(theta_df.to_string(index=False))

# --- Objective and (optional) duals ---
sw = pyo.value(model.OBJ)
print(f"\nSocial Welfare (objective) = {sw:.2f} NOK\n")

# if you enabled model.dual:
if hasattr(model, 'dual'):
    lambdas = pd.DataFrame([
        (n, model.dual[model.power_balance_const[n]])
        for n in model.N
    ], columns=['Node','Marginal Price'])
    print("=== Nodal Prices ===")
    print(lambdas.to_string(index=False))