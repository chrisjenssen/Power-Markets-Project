import pyomo as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd




def DCOPF(Data):

    # Model

    model = pyo.ConcreteModel()  # Establish the optimization model, as a concrete model

    # Sets:

    model.N = pyo.Set(ordered=True, initialize=Data["Nodes"]["NodeList"])  # Set for nodes

    # Parameters:

    model.Demand = pyo.Param(model.N, initialize=Data["Nodes"]["DEMAND"])  # Parameter for demand for every node

    #model.P_min = pyo.Param(model.N, initialize=Data["Nodes"]["GENMIN"])  # Parameter for minimum production for every node

    model.capacity = pyo.Param(model.N, initialize=Data["Nodes"]["capacity"])  # Parameter for max production for every node

    model.MC = pyo.Param(model.N, initialize=Data["Nodes"]["MC"])  # Parameter for generation cost for every node

    #model.Cost_shed = pyo.Param(initialize=Data["ShedCost"])  # Parameter for cost of shedding power

    model.Pu_base = pyo.Param(initialize=Data["pu-Base"])  # Parameter for per unit factor

    # Line capacity:

    model.DC_cap    = pyo.Param(model.H, initialize = Data["DC-lines"]["Cap"])          #Parameter for Cable capacity for every cable


    """
    Variables
    """

    # Nodes
    model.theta = pyo.Var(model.N)  # Variable for angle on bus for every node

    model.gen = pyo.Var(model.N)  # Variable for generated power on every node

    #model.shed = pyo.Var(model.N, within=pyo.NonNegativeReals)  # Variable for shed power on every node

    # DC-lines

    model.flow_DC = pyo.Var(model.H)  # Variable for power flow on every cable

    """
    Objective function: 
    """

    def ObjRule(model):  # Define objective function
        return (sum(model.gen[n] * model.Cost_gen[n] for n in model.N) + \
                sum(model.shed[n] * model.Cost_shed for n in model.N))

    model.OBJ = pyo.Objective(rule=ObjRule, sense=pyo.minimize)  # Create objective function based on given function

    """
    Constraints
    """

    """
    # Minimum generation
    # Every generating unit must provide at least the minimum capacity

    def Min_gen(model, n):
        return (model.gen[n] >= model.P_min[n])

    model.Min_gen_const = pyo.Constraint(model.N, rule=Min_gen)
    """

    # Maximum generation
    # Every generating unit cannot provide more than maximum capacity

    def Max_gen(model, n):
        return (model.gen[n] <= model.P_max[n])

    model.Max_gen_const = pyo.Constraint(model.N, rule=Max_gen)

    # Maximum from-flow line
    # Sets the higher gap of line flow from unit n

    def From_flow(model, l):
        return (model.flow_AC[l] <= model.P_AC_max[l])

    model.From_flow_L = pyo.Constraint(model.L, rule=From_flow)

    # Maximum to-flow line
    # Sets the higher gap of line flow to unit n (given as negative flow)

    def To_flow(model, l):
        return (model.flow_AC[l] >= -model.P_AC_min[l])

    model.To_flow_L = pyo.Constraint(model.L, rule=To_flow)

    # Maximum from-flow cable
    # Sets the higher gap of cable flow from unit n

    def FlowBalDC_max(model, h):
        return (model.flow_DC[h] <= model.DC_cap[h])

    model.FlowBalDC_max_const = pyo.Constraint(model.H, rule=FlowBalDC_max)

    # Maximum to-flow cable
    # Sets the higher gap of cable flow to unit n (given as negative flow)

    def FlowBalDC_min(model, h):
        return (model.flow_DC[h] >= -model.DC_cap[h])

    model.FlowBalDC_min_const = pyo.Constraint(model.H, rule=FlowBalDC_min)

    # If we want to run the model using DC Optimal Power Flow
    if Data["DCFlow"] == True:
        # Set the reference node to have a theta == 0

        def ref_node(model):
            return (model.theta[Data["Reference node"]] == 0)

        model.ref_node_const = pyo.Constraint(rule=ref_node)

        # Loadbalance; that generation meets demand, shedding, and transfer from lines and cables

        def LoadBal(model, n):
            return (model.gen[n] + model.shed[n] == model.Demand[n] + \
                    sum(Data["B-matrix"][n - 1][o - 1] * model.theta[o] * model.Pu_base for o in model.N) + \
                    sum(Data["DC-matrix"][h - 1][n - 1] * model.flow_DC[h] for h in model.H))

        model.LoadBal_const = pyo.Constraint(model.N, rule=LoadBal)

        # Flow balance; that flow in line is equal to change in phase angle multiplied with the admittance for the line

        def FlowBal(model, l):
            return (model.flow_AC[l] / model.Pu_base == (
                        (model.theta[model.AC_from[l]] - model.theta[model.AC_to[l]]) * -
                Data["B-matrix"][model.AC_from[l] - 1][model.AC_to[l] - 1]))

        model.FlowBal_const = pyo.Constraint(model.L, rule=FlowBal)

