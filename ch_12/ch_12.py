# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Branch and Cut
# I'll be experimenting with cutting planes on a mixed integer model from [miplib](https://miplib.zib.de), the [markshare2](https://miplib.zib.de/instance_details_markshare2.html) problem.
#
# ## Observations
# After running several variations of the problem I made a few observations.
# - This particular model seems to be relatively easy to get the optimal solution through heuristics/cuts/etc., but just plain needs some time to crank through nodes, reduce the upper bound, and prove optimality.
# - This doesn't mean cuts aren't helpful, however, as the run without cuts went through about 1.25 million nodes, while the run with conservative cuts went through about .22 million nodes, and the run with only the flow cover cuts about 0.09 million nodes.
# - It's hard to know however, exactly what cut types to use, and at what agression levels as, for example, the aggressive level visited more nodes than the very aggressive level, but took a little longer.
# - Setting just the particular cuts that dominate can be a good strategy as it uses the information from the cuts as best it can, but doesn't waste too much time calculating cuts that don't help it any.
# - That seems to be the main trade-off of cuts, is that they do help shorten the branch-and-bound search significantly (in terms of number of nodes), but at the expense of each node taking a more significant amount of time.
# - Like many things with MIPS, the effect of cuts and different kinds of cuts can be highly problem dependent.

# %%
import gurobipy as gp

model = gp.read("neos5.mps")
model.params.TimeLimit = 60


# %%
### Solve baseline problem to compare to
model.reset()
model.params.Cuts = -1  # Default automatic cut selection
model.optimize()
print(model.Runtime)


# %%
model.reset()
model.params.Cuts = 0  # No cuts used at all
model.optimize()
print(model.Runtime)


# %%
model.reset()
model.params.Cuts = 1  # Conservative cut generation
model.optimize()
print(model.Runtime)


# %%
model.reset()
model.params.Cuts = 2  # Agressive cut generation
model.optimize()
print(model.Runtime)


# %%
model.reset()
model.params.Cuts = 3  # Very Agressive cut generation
model.optimize()
print(model.Runtime)


# %%
model.reset()
model.params.Cuts = 0  # No cut generation

# I looked at the output of the other methods and determined that Flow Cover and
# MIR cuts seemed to be the most helpful, so I turned off all other cuts and
# just used those two on their most aggressive settings and it seemed to do well
model.params.FlowCoverCuts = 2
model.params.MIRCuts = 2
model.optimize()
print(model.Runtime)


# %%
model.reset()
model.params.Cuts = 0  # No cut generation

# And trying just Flow Cover as this seems to be the most dominant cut
model.params.FlowCoverCuts = 2
model.optimize()
print(model.Runtime)
