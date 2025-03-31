from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd

production_file = "production_data.csv"
consumption_file = "consumption_data(1).csv"

# Importing data

def ImportData(production_data_filename, consumption_data_filename):
    p_data = pd.read_csv(production_data_filename).set_index('Producer', drop=True)
    c_data = pd.read_csv(consumption_data_filename).set_index('Consumer', drop=True)
    return p_data, c_data

p_data, c_data = ImportData('production_data.csv', 'consumption_data(1).csv')


# Define the sets, and the set-dependent parameters, and variables

producers = p_data.index.tolist()
p_capacity = dict(zip(producers, p_data['Capacity'].tolist()))
p_cost = dict(zip(producers, p_data['Cost'].tolist()))
p_emission = dict(zip(producers, p_data['EmissionFactor'].tolist()))

consumers = c_data.index.tolist()
c_capacity = dict(zip(consumers, c_data['Capacity'].tolist()))
c_cost = dict(zip(consumers, c_data['Cost'].tolist()))


model = ConcreteModel()

# Define the sets
model.p = Set(initialize=producers)
model.c = Set(initialize=consumers)

# Define parameters
model.p_capacity = Param(model.p, initialize=p_capacity)
model.p_cost = Param(model.p, initialize=p_cost)
model.p_emission = Param(model.p, initialize=p_emission)
model.c_capacity = Param(model.c, initialize=c_capacity)
model.c_cost = Param(model.c, initialize=c_cost)

# Define variables
model.p_P = Var(model.p, within=NonNegativeReals) # Production per producer
model.c_C = Var(model.c, within=NonNegativeReals) # Consumption per consumer

# Mathematical formulation, maximize social surplus, constraints

def objective_rule(model):
    # consumer price* generation per generator - generator cost*generator capacity
    return sum(model.c_cost[d]*model.c_C[d] for d in model.c) - sum(model.p_cost[g]*model.p_P[g] for g in model.p)
model.objective = Objective(rule=objective_rule, sense=maximize)

# Define constraints
# Contraint: producer/generator capacity
def producer_capacity_rule(model, g):
    return model.p_P[g] <= model.p_capacity[g]
model.producer_capacity = Constraint(model.p, rule=producer_capacity_rule)

# Constraint: consumer capacity
def consumer_capacity_rule(model, d):
    return model.c_C[d] <= model.c_capacity[d]
model.consumer_capacity = Constraint(model.c, rule=consumer_capacity_rule)

# Constraint: power balance, supply = demand
def power_balance_rule(model):
    return sum(model.p_P[g] for g in model.p) == sum(model.c_C[d] for d in model.c)
model.power_balance = Constraint(rule=power_balance_rule)


# Create Pyomo model and solve using gurobi

solver = SolverFactory('gurobi')
model.dual = Suffix(direction=Suffix.IMPORT)
results = solver.solve(model, tee=True)

# Find the market price, the quantity sold/bought for each company and total surplus.
print(f"\n{'='*10} Task 1 {'='*10}")
if model.power_balance in model.dual:
    print("Electricity market clearing price: ", abs(model.dual[model.power_balance]))
else:
    print("No dual value for electricity market clearing price")

print(f"{'='*10} Optimal Solution {'='*10}")
print("Social welfare: ", model.objective())
for g in model.p:
    print(f"Producer {g} produced {model.p_P[g]()} MW")
for d in model.c:
    print(f"Consumer {d} consumed {model.c_C[d]()} MW")


