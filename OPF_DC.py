
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


from funskjoner import read_excel_file

filename = 'Problem_2_data.xlsx'
sheet_1 = 'Problem 2.2 - Base case'
sheet_2 = 'Problem 2.3 - Generators'
sheet_3 = 'Problem 2.4 - Loads'
sheet_4 = 'Problem 2.5 - Environmental'

generator, load, transmission = read_excel_file(filename, sheet_1)

""" ---- Set up the optimization model ---- """
model = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

""" ---- Sets ---- """

model.N = pyo.Set(ordered = True, initialize = load['Location'].unique().tolist())   #Set for nodes

model.G = pyo.Set(ordered=True, initialize=generator['Generator'].tolist())

model.T = pyo.Set(ordered = True, initialize = transmission['Line'].tolist())  #Set for transmission lines

""" ---- Parameters ----"""

#Nodes

model.Demand    = pyo.Param(model.N, initialize = load.set_index('Load unit')['Demand [MW]'].to_dict())  #Parameter for demand for every node

#Generators

model.Capacity_gen = pyo.Param(model.G, initialize=generator.set_index('Generator')['Capacity [MW]'].to_dict())

model.MarginalCost = pyo.Param(model.G, initialize=generator.set_index('Generator')['Marginal cost NOK/MWh]'].to_dict())

#model.Pu_base   = pyo.Param(initialize = Data["pu-Base"])                   #Parameter for per unit factor

#Transmission lines

model.Capacity_trans  = pyo.Param(model.T, initialize = transmission["Capacity [MW]"].to_dict())     #Parameter for max transfer from node, for every line


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

#Set the reference node to have a theta == 0

def ref_node(model):
    return(model.theta[Data["Reference node"]] == 0)
model.ref_node_const = pyo.Constraint(rule = ref_node)


# Power balance constraint; that generation meets demand, shedding, and transfer from lines and cables
def power_balance_rule(model, n):
    return (sum(model.gen[g] for g in model.G if model.generator_location[g] == n) ==
            model.Demand[n] +
            sum(model.flow_trans[t] for t in model.T if model.transmission_to[t] == n) -
            sum(model.flow_trans[t] for t in model.T if model.transmission_from[t] == n))

model.power_balance_const = pyo.Constraint(model.N, rule=power_balance_rule)


# Flow balance constraint
def flow_balance_rule(model, t):
    from_node = model.transmission_from[t]
    to_node = model.transmission_to[t]
    return model.flow_trans[t] == model.susceptance[t] * (model.theta[from_node] - model.theta[to_node])

model.flow_balance_const = pyo.Constraint(model.T, rule=flow_balance_rule)
 
    
    
"""
Compute the optimization problem
"""
    
#Set the solver for this
#opt         = SolverFactory("glpk")
opt         = SolverFactory('gurobi',solver_io="python")



#Enable dual variable reading -> important for dual values of results
model.dual      = pyo.Suffix(direction=pyo.Suffix.IMPORT)


#Solve the problem
results     = opt.solve(model, load_solutions = True)

#Write result on performance
results.write(num=1)

    
