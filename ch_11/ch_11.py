# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Branch and Bound Implementation
# I'm going to use the book's Example 11.2 problem (which is a MIP) for testing the implementation. This means I'll follow the convention of "best-first" node selection and "lowest-index" branching variable selection. I will, however, attempt to code it so that those can be easily changed.
# ## The Example Problem
# $$
# \begin{align*}
#     \max &\ z = -y_1 + 2y_2 + y_3 + 2x_1 \\
#     \text{s.t.} &\ y_1 + y_2 - y_3 + 3x_1 \le 7 \\
#     & y_2 + 3y_3 - x_1 \le 5 \\
#     & 3y_1 + x_1 \ge 2 \\
# \end{align*}
# $$

# %%
import gurobipy as gp
from collections import Iterable
import heapq
from math import ceil, floor


# Build Base Model
m = gp.Model()
m.params.LogToConsole = 0
m.ModelSense = -1  # Maximization
INT = gp.GRB.INTEGER
CONT = gp.GRB.CONTINUOUS
var_types = [INT] * 3 + [CONT]
var_keys = ["y1", "y2", "y3", "x"]
variables = m.addVars(
    var_keys,
    lb=0.0,
    ub=float("inf"),
    obj=(-1, 2, 1, 2),
    vtype=var_types,
    name=var_keys,
)
m.addConstr(
    variables["y1"] + variables["y2"] - variables["y3"] + 3 * variables["x"] <= 7
)
m.addConstr(variables["y2"] + 3 * variables["y3"] - variables["x"] <= 5)
m.addConstr(3 * variables["y1"] + variables["x"] >= 2)
m.update()  # Ensure that the model is up-to-date all the way.

# %% [markdown]
# # Branch and Bound
# Some definitions: I call the best possible bound the *optimistic* bound. For a maximization problem this is the least upper bound or $\bar{z}$ from the book. I call the best incumbent bound the *feasible* bound. For a maximization problem this is the greatest lower bound or $\underline{z}$ from the book. This is so that my terminology can (hopefully) be more easily applied to either minimization or maximization without confusion.
#
# 1. Initialize: Solve the LP relaxation of original problem. Update globals variables and add to queue.
# 2. Choose a Node: If the queue is empty, end; the incumbent solution is optimal. Otherwise, select a node (pop from priority/fifo queue, etc.).
# 3. Update Optimistic (Upper) Bound: Solve the node and set the optimistic bound of the subproblem.
# 4. Prune if Infeasible: If the LP is infeasible prune and go to Step 2. Otherwise, continue.
# 5. Prune by Bound: If the objective value of the LP is worse than the feasible bound, prune and go to Step 2. Otherwise, continue.
# 6. Prune by Optimality:
#    1. If the LP is integer feasible: If its objective value is better than the feasible bound, update the feasible bound and incumbent solutions and then prune the node. If not, just prune the node. Either way, go to Step 2.
#    2. Otherwise: Continue.
# 7. Branching: Choose a variable to branch on and create the two nodes that branch off that variable. Add these to the queue.

# %%
# Step 1: Solve the root relaxation


def are_not_integer(solution, vtypes, epsilon=1e-5):
    return map(
        lambda x: x[0],
        filter(
            lambda x: x[1][1] == INT and abs(round(x[1][0]) - x[1][0]) >= epsilon,
            enumerate(zip(solution, vtypes)),
        ),
    )


def int_feasible(variables: Iterable, vtypes: list[str]):
    return not any(True for _ in are_not_integer([var.X for var in variables], vtypes))


def initialize(root: gp.Model):
    nodes = [{"lp": root, "optimistic": float("inf"), "parent_idx": None}]
    queue = []
    heapq.heappush(queue, priority(float("inf"), 0))
    return nodes, queue


# %%
def priority(optimistic_bound, creation_index):
    # Results in best-bound with fifo breaking ties
    return (-optimistic_bound, creation_index)


def get_node_idx(priority):
    # Get the node index from the info stored in the queue
    return priority[1]


def select_branching_variable(solution, vtypes):
    # Results in a lowest-index selection
    return next(are_not_integer(solution, vtypes))


def lp(node) -> gp.Model:
    return node["lp"]


def add_node(node, nodes, queue):
    node_idx = len(nodes)
    nodes.append(node)
    heapq.heappush(queue, priority(lp(nodes[node["parent_idx"]]).ObjVal, node_idx))


def get_next_node(nodes, queue):
    next_idx = get_node_idx(heapq.heappop(queue))
    return nodes[next_idx], next_idx


def get_solution(node):
    return [var.X for var in lp(node).getVars()]


def prune_infeasibility(node):
    return lp(node).Status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD, gp.GRB.UNBOUNDED]


def prune_bound(node, feasible_bound):
    return lp(node).ObjVal <= feasible_bound


def prune_integer(node, vtypes):
    return int_feasible(lp(node).getVars(), vtypes)


def create_children(node, node_idx, vtypes):
    solution = get_solution(node)
    branch_idx = select_branching_variable(solution, vtypes)
    branch_val = solution[branch_idx]
    constraint_funcs = [
        lambda x: x <= floor(branch_val),
        lambda x: x >= ceil(branch_val),
    ]

    children = []
    for func in constraint_funcs:
        print(
            f"\tCreating Child from {node_idx + 1} with constraint y{branch_idx + 1} "
            f"{f'<= {floor(branch_val)}' if func is constraint_funcs[0] else f'>= {ceil(branch_val)}'}"
        )
        child_lp = lp(node).copy()
        child_vars = child_lp.getVars()
        child_lp.addConstr(func(child_vars[branch_idx]))
        children.append(
            {"lp": child_lp, "optimistic": lp(node).ObjVal, "parent_idx": node_idx}
        )
    return children


def branch_and_bound(m: gp.Model):
    vtypes = [var.vtype for var in m.getVars()]
    root = m.relax()
    root.update()

    feasible_bound = -float("inf")
    solutions = []

    nodes, queue = initialize(root)
    k = 1
    while queue:
        node, node_idx = get_next_node(nodes, queue)
        print(f"Visiting node {node_idx + 1} at iteration {k}:")
        node["lp"].optimize()
        k += 1
        if prune_infeasibility(node):
            print("\tPruned by infeasibility")
            continue
        print(f"\tSolution: {get_solution(node)}")
        if prune_bound(node, feasible_bound):
            print(f"\tPruned by bound: {lp(node).ObjVal} < {feasible_bound}")
            continue
        if prune_integer(node, vtypes):
            print(f"\tPruned by integer solution: {lp(node).ObjVal} ({feasible_bound})")
            if lp(node).ObjVal > feasible_bound:
                feasible_bound = node["lp"].ObjVal
                solutions.append(get_solution(node))
            continue

        children = create_children(node, node_idx, vtypes)
        for child in children:
            add_node(child, nodes, queue)

    return solutions[-1]


solution = branch_and_bound(m)
print(f"Final Solution: {solution}")
