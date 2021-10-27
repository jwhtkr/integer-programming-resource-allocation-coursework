# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Dantzig-Wolfe Decomposition
# In this assignment I'll be using the Dantzig-Wolfe Decomposition method to solve LPs.

# %%
import gurobipy as gp
import numpy as np

# %% [markdown]
# ## Assumptions
# I'm assuming that the LP constraints are decomposed into two sets. The "normal" ones are in the matrix A and vector b, and the "special" ones are in the matrix G and vector d.
# Additionally, all constraints are assumed to be less than or equal constraints. The problem is also assumed to be feasible and bounded (i.e., "nice"). The initial solution of $x = \mathbf{0}$ is assumed to be feasible. The partitioned region defined by G and d is assumed to be bounded.
#
# Note that all of these assumptions are addressed and removed in a full Dantzig-Wolfe decomposition method (i.e., multiple constraint partitions, mixed constraints, infeasible and unbounded problems, finding initial solutions, and partially bounded partitioned constraint regions), but are used here to simplify the problem to the main decomposition structure.

# %%
def form_initial_master_tableau(c, A, b, G, d):
    initial_basis = np.eye(A.shape[0] + 1)  # B matrix
    tableau = np.vstack(
        [
            np.hstack([np.zeros((A.shape[0] + 1)), [0.0]]),
            np.hstack([initial_basis, np.concatenate([b, [1.0]]).reshape(-1, 1)]),
        ]
    )
    return (
        tableau,
        {**{i: [] for i in range(A.shape[0])}, **{A.shape[0]: [0]}},
        np.zeros((len(c),)),
    )


def construct_subproblem(master_tableau, A, c, G, d):
    subproblem = gp.Model()
    u = master_tableau[0, :-2]
    alpha = master_tableau[0, -2]
    x = subproblem.addMVar((len(c),))
    print(f"Subproblem objective coefficients:\n{u @ A - c}\n")
    subproblem.setObjective((u @ A - c) @ x + alpha)
    subproblem.addConstr(G @ x <= d)
    return subproblem


def generate_column(subproblem, master_tableau, A):
    subproblem.optimize()
    c_bar = subproblem.getObjective().getValue()
    x = np.array([var.X for var in subproblem.getVars()])
    print(f"Subproblem solution:\n{x}")
    a_bar = master_tableau[1:, :-1] @ np.concatenate([A @ x, [1.0]])
    print(f"New column:\nc_bar: {c_bar}\na_bar:\n{a_bar}\n")
    return (c_bar, a_bar), x


def optimal(new_column):
    return new_column[0] >= 0


def update_master(master_tableau, new_column):
    x = master_tableau[1:, -1]
    c_bar, a_bar = new_column
    leaving_row_idx = np.argmin(x / np.ma.masked_less_equal(a_bar, 0.0)) + 1
    augmented_tableau = np.hstack(
        [master_tableau, np.vstack([[c_bar], a_bar.reshape(-1, 1)])]
    )
    augmented_tableau[leaving_row_idx, :] /= augmented_tableau[leaving_row_idx, -1]
    leaving_row = augmented_tableau[leaving_row_idx, :]
    print(f"Leaving Row: {leaving_row_idx}")

    for row_idx, row in enumerate(augmented_tableau):
        if row_idx == leaving_row_idx:
            continue
        row -= leaving_row * row[-1]

    return augmented_tableau[:, :-1], leaving_row_idx - 1


def solution(master_tableau, row_map, subproblem_solutions):
    solution = subproblem_solutions[0]  # should be the zero solution
    for row_idx, subproblem_idxs in row_map.items():
        if subproblem_idxs:
            solution += (
                master_tableau[row_idx + 1, -1]
                * subproblem_solutions[subproblem_idxs[-1]]
            )
    return solution


def solve_dantzig_wolfe(c, A, b, G, d):
    master_tableaus = []
    subproblem_solutions = []
    tableau, row_map, initial_solution = form_initial_master_tableau(c, A, b, G, d)
    master_tableaus.append(tableau)
    subproblem_solutions.append(initial_solution)
    is_optimal = False

    i = 0
    while not is_optimal:
        print(f"Current Tableau:\n{master_tableaus[-1]}")
        subproblem = construct_subproblem(master_tableaus[-1], A, c, G, d)
        new_column, subproblem_solution = generate_column(
            subproblem, master_tableaus[-1], A
        )
        subproblem_solutions.append(subproblem_solution)
        i += 1
        is_optimal = optimal(new_column)
        print(f"Is Optimal? {is_optimal}")
        tableau, changed_row_idx = update_master(master_tableaus[-1], new_column)
        row_map[changed_row_idx].append(i)
        master_tableaus.append(tableau)

    return solution(master_tableaus[-1], row_map, subproblem_solutions)


# %%
c = np.array([2, 2, 3, -1])
A = np.array([[1, 1, 1, 2], [-2, 2, 1, 1]])
b = np.array([17, 11])
G = np.array([[-1, 0, 0, 1], [2, 0, 1, 0], [0, 1, 0, 1]])
d = np.array([2, 9, 5])

sol = solve_dantzig_wolfe(c, A, b, G, d)
print(f"Solution:\n{sol}")
