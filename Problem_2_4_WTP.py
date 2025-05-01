import pyomo.environ as pyo
from pyomo.core import display
from pyomo.opt import SolverFactory
import pandas as pd

def read_excel_file_2_4_WTP(fileName, sheetName):
    # Read everything in
    df = pd.read_excel(fileName, sheet_name=sheetName, header=2)

    # ---- Generators ----
    gen = df.loc[:, 'Generator':'Slack bus'].copy()
    # throw away any row that didn’t really have a generator
    gen = gen[gen['Generator'].notna()]

    # ---- Loads ----
    ld = df.loc[:, 'Load unit':'Location.1'].copy()
    ld = ld[ld['Load unit'].notna()]

    # ensure WTP is numeric, fill missing with 0.0
    ld['Marginal WTP [NOK/MWh]'] = (
        pd.to_numeric(ld['Marginal WTP [NOK/MWh]'], errors='coerce')
        .fillna(0.0))

    # ---- Transmission lines ----
    tr = df.loc[:, 'Line':'Susceptance [p.u]'].copy()
    tr = tr[tr['Line'].notna()]

    return gen, ld, tr

def OPF_DC_2_4_WTP(generator, load, transmission):
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

    # ---- NEW: define a load‐unit set L ----
    model.L = pyo.Set(initialize=load['Load unit'].tolist())

    # map each load → its node
    model.LoadLocation = pyo.Param(
        model.L,
        initialize=load.set_index('Load unit')['Location.1'].to_dict()
    )
    # demand and WTP per load
    model.D_L = pyo.Param(
        model.L,
        initialize=load.set_index('Load unit')['Demand [MW]'].to_dict()
    )
    model.WTP = pyo.Param(
        model.L,
        initialize=load.set_index('Load unit')['Marginal WTP [NOK/MWh]'].to_dict()
    )

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

    # ---- replace fixed-demands by served‐load vars ----
    # how much of each load‐unit we actually serve (0 ≤ serve ≤ D_L)
    model.serve = pyo.Var(model.L, bounds=lambda M, ℓ: (0, M.D_L[ℓ]), initialize=lambda M, ℓ: M.D_L[ℓ])

    # force inelastic loads (WTP=0) to be fully served
    def inelastic_rule(M, ℓ):
        if M.WTP[ℓ] == 0.0:
            return M.serve[ℓ] == M.D_L[ℓ]
        return pyo.Constraint.Skip

    model.inelastic_const = pyo.Constraint(model.L, rule=inelastic_rule)

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
    Maximize social welfare
    """

    model.OBJ = pyo.Objective(
        expr=sum(model.WTP[ℓ] * model.serve[ℓ] for ℓ in model.L)
             - sum(model.MarginalCost[g] * model.gen[g] for g in model.G),
        sense=pyo.maximize
    )

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

    # ---- revised power‐balance: gens + inflow = Σ serve + outflow ----
    def balance_rule(M, n):
        gen_sum = sum(M.gen[g] for g in M.G if M.GenLocation[g] == n)
        inflow = sum(M.flow_trans[t] for t in M.T if M.transmission_to[t] == n)
        outflow = sum(M.flow_trans[t] for t in M.T if M.transmission_from[t] == n)
        load_sum = sum(M.serve[ℓ] for ℓ in M.L if M.LoadLocation[ℓ] == n)
        return gen_sum + inflow == load_sum + outflow

    model.power_balance_const = pyo.Constraint(model.N, rule=balance_rule)

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
