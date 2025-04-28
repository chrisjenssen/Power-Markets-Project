import pyomo.environ as pyo
from pyomo.core import display
from pyomo.opt import SolverFactory
import pandas as pd


def OPF_DC_2_4(generator, load, transmission):
    """
    This function sets up and solves a DC Optimal Power Flow (OPF) problem using Pyomo.
    It takes in generator, load, and transmission data as input.
    """

    """ ---- Set up the optimization model ---- """
    model = pyo.ConcreteModel()  # Establish the optimization model, as a concrete model in this case

    """ ---- Sets ---- """

    model.N = pyo.Set(ordered=True, initialize=load['Location.1'].unique().tolist())  # Set for nodes

    model.G = pyo.Set(ordered=True, initialize=generator['Generator'].tolist())

    # map each generator → its node
    model.GenLocation = pyo.Param(
        model.G,
        initialize=generator.set_index('Generator')['Location'].to_dict()
    )

    model.T = pyo.Set(ordered=True, initialize=transmission['Line'].tolist())  # Set for transmission lines

    """ ---- Parameters ----"""

    # Nodes

    # build a dict: node → total demand (sum over all load units at that node)
    demand_by_node = (
        load
        .groupby('Location.1')['Demand [MW]']
        .sum()
        .to_dict()
    )
    model.Demand = pyo.Param(model.N, initialize=demand_by_node)  # Parameter for demand for every node

    # Generators

    model.Capacity_gen = pyo.Param(model.G, initialize=generator.set_index('Generator')['Capacity [MW]'].to_dict())

    model.MarginalCost = pyo.Param(model.G,
                                   initialize=generator.set_index('Generator')['Marginal cost NOK/MWh]'].to_dict())

    model.Pu_base = pyo.Param(initialize=1000)  # Parameter for per unit factor

    # Transmission lines

    model.Capacity_trans = pyo.Param(model.T, initialize=transmission.set_index('Line')["Capacity [MW].1"].to_dict())

    model.Susceptance = pyo.Param(model.T, initialize=transmission.set_index('Line')[
        "Susceptance [p.u]"].to_dict())  # Parameter for max transfer from node, for every line

    model.GenLocation = pyo.Param(model.G, initialize=generator.set_index('Generator')['Location'].to_dict())

    # Build from/to node mappings for each line
    # this splits the transmission DataFrame 'Line' column at "-" "Line 1-2"
    splits = transmission['Line'].str.extract(r'Line\s*(\d+)-(\d+)')
    transmission['from_node'] = 'Node ' + splits[0]
    transmission['to_node'] = 'Node ' + splits[1]

    # Add those mappings as Pyomo Params

    model.transmission_from = pyo.Param(model.T, initialize=transmission.set_index('Line')['from_node'].to_dict())

    model.transmission_to = pyo.Param(model.T, initialize=transmission.set_index('Line')['to_node'].to_dict())

    """ ---- Variables ---- """

    # Generation levels for each generator
    model.gen = pyo.Var(model.G, initialize=0.0)

    # Power flow through each transmission line
    model.flow_trans = pyo.Var(model.T, initialize=0.0)

    # Voltage angle at each node
    model.theta = pyo.Var(model.N, initialize=0.0)

    # ----> Optional variables <----#
    # Binary Variables, to model on/off decisions (e.g., whether a generator is running)
    # model.gen_status = pyo.Var(model.G, within=pyo.Binary)

    # Load shedding at each node (if applicable, dont know if it is)
    # model.shed = pyo.Var(model.N, within=pyo.NonNegativeReals)

    """
    Objective function:
    Minimize generation cost
    """

    model.OBJ = pyo.Objective(
        expr=sum(model.gen[g] * model.MarginalCost[g] for g in model.G),
        sense=pyo.minimize)

    """
    Constraints
    """

    # Minimum generation constraint
    def min_gen_rule(model, g):
        return model.gen[g] >= 0

    model.min_gen_const = pyo.Constraint(model.G, rule=min_gen_rule)

    # Maximum generation constraint
    def max_gen_rule(model, g):
        return model.gen[g] <= model.Capacity_gen[g]  # * model.gen_status[g]

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
    #reference_node = generator.loc[generator['Slack bus'], 'Location'].item()

    reference_node = generator.loc[generator['Slack bus'] == True, 'Location'].unique().tolist()

    model.theta[reference_node].fix(0)

    # Power balance at each node
    def power_balance_rule(model, n):
        # sum of gens at node n
        gen_sum = sum(model.gen[g]
                      for g in model.G
                      if model.GenLocation[g] == n)
        # inflow / outflow unchanged
        inflow = sum(model.flow_trans[t]
                     for t in model.T
                     if model.transmission_to[t] == n)
        outflow = sum(model.flow_trans[t]
                      for t in model.T
                      if model.transmission_from[t] == n)

        return gen_sum + inflow == model.Demand[n] + outflow

    model.power_balance_const = pyo.Constraint(model.N, rule=power_balance_rule)

    # DC flow law on each line
    def flow_balance_rule(model, t):
        from_n = model.transmission_from[t]
        to_n = model.transmission_to[t]
        # B_ij*(θ_i − θ_j)
        return model.flow_trans[t] == model.Susceptance[t] * (model.theta[from_n] - model.theta[to_n])

    model.flow_balance_const = pyo.Constraint(model.T, rule=flow_balance_rule)

    # model.pprint()

    """
    Compute the optimization problem
    """

    #model = pyo.ConcreteModel()

    # register dual suffix so we can pull shadow prices back in
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # solve
    opt = SolverFactory('gurobi', solver_io='python')
    results = opt.solve(model, tee=True)
    model.solutions.load_from(results)

    print("\n=== Binding constraints ===")
    for g in model.G:
        if abs(model.gen[g].value - model.Capacity_gen[g]) < 1e-6:
            print(f" Generator {g} at upper limit")
    for t in model.T:
        if abs(abs(model.flow_trans[t].value) - model.Capacity_trans[t]) < 1e-6:
            print(f" Line {t} congested")

    # 1) PRODUCTION COST per generator
    prod_data = []
    for g in model.G:
        q = model.gen[g].value
        c = model.MarginalCost[g]
        prod_data.append({
            'Generator': g,
            'Quantity [MW]': q,
            'Marginal Cost [NOK/MWh]': c,
            'Total Cost [NOK/h]': q * c
        })
    df_prod = pd.DataFrame(prod_data)

    # 2) SHADOW PRICES
    #  2a) generator limits
    gen_shadow = []
    for g in model.G:
        π_min = model.dual[model.min_gen_const[g]]  # should almost always be zero
        π_max = model.dual[model.max_gen_const[g]]
        gen_shadow.append({
            'Generator': g,
            'Dual min-gen (lower bound)': π_min,
            'Dual max-gen (upper bound)': π_max
        })
    df_gen_shadow = pd.DataFrame(gen_shadow)

    #  2b) nodal prices = dual of power balance
    node_shadow = []
    for n in model.N:
        λ = model.dual[model.power_balance_const[n]]
        node_shadow.append({'Node': n, 'LMP [NOK/MWh]': λ})
    df_node_shadow = pd.DataFrame(node_shadow)

    #  2c) line‐flow shadows
    line_shadow = []
    for t in model.T:
        μ_up = model.dual[model.max_flow_trans_const[t]]
        μ_down = model.dual[model.min_flow_trans_const[t]]
        line_shadow.append({
            'Line': t,
            'Dual max-flow (upper)': μ_up,
            'Dual min-flow (lower)': μ_down
        })
    df_line_shadow = pd.DataFrame(line_shadow)

    # Print or return them
    print("\n=== Production Costs ===")
    print(df_prod.to_string(index=False))

    print("\n=== Generator‐limit Shadow Prices ===")
    print(df_gen_shadow.to_string(index=False))

    print("\n=== Nodal Prices (LMPs) ===")
    print(df_node_shadow.to_string(index=False))

    print("\n=== Line‐flow Shadow Prices ===")
    print(df_line_shadow.to_string(index=False))

    def dump_all_vars(m):
        for varobj in m.component_objects(pyo.Var, active=True):
            print(f"\nVariable: {varobj.name}")
            for idx in varobj:
                val = varobj[idx].value
                if val is None:
                    print(f"  {idx} : n/a")
                else:
                    print(f"  {idx} : {val:.3f}" if isinstance(val, (int, float)) else f"  {idx} : {val}")

    dump_all_vars(model)


filename = 'Problem_2_data.xlsx'

sheet_1 = 'Problem 2.2 - Base case'
sheet_2 = 'Problem 2.3 - Generators'
sheet_3 = 'Problem 2.4 - Loads'
sheet_4 = 'Problem 2.5 - Environmental'

generator, load, transmission = read_excel_file(filename, sheet_3)
print(generator)
print(load)
print(transmission)


#num_buses = 3  # Set the number of buses in your system, have to do this manually due to the set up in excel file
#Y_bus = create_y_matrix(num_buses, transmission)
#Y_DC = generate_Y_DC(generator, Y_bus)

OPF_DC(generator, load, transmission)

