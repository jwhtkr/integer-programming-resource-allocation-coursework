# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import gurobipy as gp
import numpy as np
rng = np.random.default_rng(42)

# %% [markdown]
# # Knapsack Problem
# $$
# \begin{align*}
#     \max\ &z = \sum_j c_j y_j \\
#     \text{s.t.} &\sum_j a_j y_j \le b \\
#     &y_j = 0 \text{ or } 1 & j = 1, 2, \dots, n
# \end{align*}
# $$

# %%
knapsack = gp.Model()
n_var = 10

x = [knapsack.addVar(vtype=gp.GRB.BINARY) for j in range(n_var)]

c = [rng.uniform(0, 10) for j in range(n_var)]
objective = gp.quicksum(c[j]*x[j] for j in range(n_var))

a = [rng.uniform(0, 10) for j in range(n_var)]
b = rng.uniform(n_var, 3*n_var)
constraint = gp.quicksum(a[j]*x[j] for j in range(n_var)) <= b

knapsack.setObjective(objective, gp.GRB.MAXIMIZE)
knapsack.addConstr(constraint)

knapsack.optimize()

print(f"Solution vector: {[var.X for var in x]}")

# %% [markdown]
# # Capacitated Lot Sizing
# ## Decision Variables
# $y_t$: whether or not to produce in a time period.
# $x_t$: how much to produce in a time period.
# ## Slack Variables
# $s_t$: inventory level at the end of each time period ($s_0 = 0$).
# ## Input Variables
# $d_t$: demand in each time period.
# $f_t$: fixed setup cost in a time period.
# $c_t$: unit production cost.
# $h_t$: unit holding cost.
# $u_t$: production capacity for a time period.
# $$
# \begin{align*}
#     \min\ &\sum_t (c_t x_t + f_t y_t + h_t s_t) \\
#     \text{s.t.} &s_{t-1} + x_t + s_t = d_t \\
#     &x_t \le u_t y_t \\
#     &x_t \ge 0 \\
#     &s_t \ge 0 \\
#     &y_t = 0 \text{ or } 1 \\
# \end{align*}
# $$

# %%
cap_lot_size = gp.Model()
n_time_step = 10

y = [cap_lot_size.addVar(vtype=gp.GRB.BINARY) for t in range(n_time_step)]
x = [cap_lot_size.addVar(vtype=gp.GRB.CONTINUOUS) for t in range(n_time_step)]
s = [cap_lot_size.addVar(vtype=gp.GRB.CONTINUOUS) for t in range(n_time_step)]

c = [rng.uniform(0, 10) for t in range(n_time_step)]
f = [rng.uniform(0, 10) for t in range(n_time_step)]
h = [rng.uniform(0, 10) for t in range(n_time_step)]
d = [rng.uniform(0, 10) for t in range(n_time_step)]
u = [rng.uniform(5, 10) for t in range(n_time_step)]

objective = gp.quicksum(c[t]*x[t]+f[t]*y[t]+h[t]*s[t] for t in range(n_time_step))
cap_lot_size.setObjective(objective, gp.GRB.MINIMIZE)

demand_lhs = [x[0] - s[0]] + [s[t-1] + x[t] - s[t] for t in range(1, n_time_step)]
demand_constraints = [demand_lhs[t] == d[t] for t in range(n_time_step)]
capacity_constraints = [x[t] <= u[t]*y[t] for t in range(n_time_step)]
constraints = demand_constraints + capacity_constraints
for constraint in constraints:
    cap_lot_size.addConstr(constraint)

cap_lot_size.optimize()

print(f"Lot Sizes: {[var.X for var in x]}")
print(f"Surplus Sizes: {[var.X for var in s]}")


