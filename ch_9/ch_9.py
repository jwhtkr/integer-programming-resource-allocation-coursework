# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import gurobipy as gp
gp.setParam("DisplayInterval", 1)
model = gp.read("adlittle.mps")


# %%
model.params.Method = 0  # Primal Simplex
model.optimize()
print([var.X for var in model.getVars()])


# %%
model.params.Method = 1  # Dual Simplex
model.reset()
model.optimize()
print([var.X for var in model.getVars()])


