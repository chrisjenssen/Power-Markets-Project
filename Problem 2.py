from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd

def readexcelfile(fileName, sheetname):
    df = pd.read_excel(fileName,sheet_name=sheetname, header=2)
    generator_data = df.loc[:,'Generator':'Slack bus']
    load_data = df.loc[:,'Load unit':'Location']
    transmission_data = df.loc[:,'Line':'Susceptance [p.u]']
    return(generator_data, load_data, transmission_data)

fileName = 'Problem_2_data.xlsx'
sheetname = 'Problem 2.2 - Base case'

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


