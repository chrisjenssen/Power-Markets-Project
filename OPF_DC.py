
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd


def OPF_DC(generator, load, transmission):
    """
    This function sets up and solves a DC Optimal Power Flow (OPF) problem using Pyomo.
    It takes in generator, load, and transmission data as input.
    """

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

    model.GenLocation = pyo.Param(model.G, initialize = generator.set_index('Generator')['Location'].to_dict())

    # Build from/to node mappings for each line 
    # this splits the transmission DataFrame 'Line' column at "-" "Line 1-2"
    splits = transmission['Line'].str.extract(r'Line\s*(\d+)-(\d+)')
    transmission['from_node'] = 'Node ' + splits[0]
    transmission['to_node']   = 'Node ' + splits[1]

    # Add those mappings as Pyomo Params

    model.transmission_from = pyo.Param(model.T, initialize=transmission.set_index('Line')['from_node'].to_dict())
    
    model.transmission_to   = pyo.Param(model.T, initialize=transmission.set_index('Line')['to_node'].to_dict())


    """ ---- Variables ---- """

    # Generation levels for each generator
    model.gen = pyo.Var(model.G, initialize=0.0)
    
    # Power flow through each transmission line
    model.flow_trans = pyo.Var(model.T, initialize=0.0)

    # Voltage angle at each node
    model.theta = pyo.Var(model.N,initialize=0.0)

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    #----> Optional variables <----#
    #Binary Variables, to model on/off decisions (e.g., whether a generator is running)
    #model.gen_status = pyo.Var(model.G, within=pyo.Binary)

    # Load shedding at each node (if applicable, dont know if it is)
    #model.shed = pyo.Var(model.N, within=pyo.NonNegativeReals)

    """
    Objective function:
    Minimize generation cost
    """

    model.OBJ = pyo.Objective(
        expr = sum(model.gen[g] * model.MarginalCost[g] for g in model.G),
        sense = pyo.minimize)


    """
    Constraints
    """
    # Minimum generation constraint
    def min_gen_rule(model, g):
        return model.gen[g] >= 0

    model.min_gen_const = pyo.Constraint(model.G, rule=min_gen_rule)

    # Maximum generation constraint
    def max_gen_rule(model, g):
        return model.gen[g] <= model.Capacity_gen[g] #* model.gen_status[g]

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
    reference_node = generator.loc[generator['Slack bus'], 'Location'].item()
    model.theta[reference_node].fix(0)

    # Power balance at each node
    def power_balance_rule(model, n):
        # total generation at node n
        gen_sum = sum(model.gen[g] for g in model.G if generator.set_index('Generator')['Location'][g] == n)
        # total inflow to n
        inflow = sum(model.flow_trans[t] for t in model.T if model.transmission_to[t] == n)
        # total outflow from n
        outflow = sum(model.flow_trans[t] for t in model.T if model.transmission_from[t] == n)
        # balance: generation + inflow == demand + outflow
        return gen_sum + inflow == model.Demand[n] + outflow

    model.power_balance_const = pyo.Constraint(model.N, rule=power_balance_rule)


    # DC flow law on each line
    def flow_balance_rule(model, t):
        from_n = model.transmission_from[t]
        to_n   = model.transmission_to[t]
        # B_ij*(θ_i − θ_j)
        return model.flow_trans[t] == model.Susceptance[t] * (model.theta[from_n] - model.theta[to_n])

    model.flow_balance_const = pyo.Constraint(model.T, rule=flow_balance_rule)

    #model.pprint()

    """
    Compute the optimization problem
    """

    opt = SolverFactory('gurobi', solver_io='python')
    # tee=True will stream the solver log to your console
    results = opt.solve(model, tee=True)

    # For some Pyomo solvers you need to explicitly load the solution
    model.solutions.load_from(results)

    # Write result on performance
    results.write()

    #results = opt.solve(model, tee=True)
    print("Status:", results.solver.status)
    print("Termination:", results.solver.termination_condition)

    def dump_all_vars(m):
        for varobj in m.component_objects(pyo.Var, active=True):
            print(f"\nVariable: {varobj.name}")
            for idx in varobj:
                val = varobj[idx].value
                if val is None:
                    print(f"  {idx} : n/a")
                else:
                    print(f"  {idx} : {val:.3f}" if isinstance(val, (int,float)) else f"  {idx} : {val}")

    # usage
    dump_all_vars(model)

    # Extract and print shadow prices (dual variables)
    print("\nShadow Prices (Dual Variables):")

    for const in model.component_objects(pyo.Constraint, active=True):
        print(f"\nConstraint: {const.name}")
        for idx in const:
            try:
                dual_value = model.dual[const[idx]]
                if dual_value is not None:
                    print(f"  {idx} : {dual_value:.3f}")
                else:
                    print(f"  {idx} : n/a")
            except KeyError:
                print(f"  {idx} : Dual value not available")


"""
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
"""