#%%
from gurobipy import Model, GRB
import os
from copy import copy
import numpy as np

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# %%
#Initialize model
def base_model():
    model = Model()

    #Set Variables
    x = model.addVars(3, vtype="C", lb = 0.0, name = "edge")
    p = model.addVars(4, vtype="C", lb = 0.0, name ="path")

    #Set Objective
    model.setObjective(x[0] * (x[0] + 1) + 3 * x[1]**2 + x[2] * (x[2] + 1), GRB.MINIMIZE)

    model.addConstr(p[0] + p[1] == 1, name="top_flow")
    model.addConstr(p[2] + p[3] == 1, name="botom_flow")
    model.addConstr(x[0] == p[0], name="top_edge")
    model.addConstr(x[1] == p[1] + p[2], name="middle_edge")
    model.addConstr(x[2] == p[3], name="bottom_edge")
    return model, x, p


def print_res(model, x, p):
    print(f'''
Flow through top edge is {round(x[0].X, 3)}.
Flow through middle edge is {round(x[1].X, 3)}.
Flow through bottom edge is {round(x[2].X, 3)}.
Top path has flow of {round(p[0].X, 3)}
Top middle path has flow of {round(p[1].X, 3)}
Bottom middle path has flow of {round(p[2].X, 3)}
Bottom path has flow of {round(p[3].X, 3)}
Cost of this traffic pattern is {round(model.objVal, 3)}
''')

model, x, p = base_model()

#Write our model to check it over
model.write("social_opt.lp")

#%%
#Optimize
model.optimize()
#%%
#Print Results
print("Social Optimum: \n")
print_res(model, x, p)
# %%

model2, x2, p2 = base_model()

model2.addConstr(1 + x2[0] == 3*x2[1], name="agent_1")
model2.addConstr(1 + x2[2] == 3*x2[1], name="agent_2")

model2.write("ge_notolls.lp")
model2.optimize()

#%%
print("General Equilibrium: \n")
print_res(model2, x2, p2)

#%%

model3, x3, p3 = base_model()

#Solved by solving the system of equations:
#c1 + 1 + x1 = 3x2 + c2 = c3 + 1+ x3  <=> All path choices' costs for each agent are equal (symmetry is property of network)
#c1x1 + c2x2 + c3x3 = 0 <=> Ensures a balance budget between tolls and subsidies

c1 = x[1].X * (3 * x[1].X - 1 - x[0].X) /(2*x[0].X + x[1].X)
c2 = -2 * c1 * x[0].X / x[1].X
c3 = c1

model3.addConstr(1 + x3[0] + c1 ==  c2 + 3*x3[1], name="agent_1")
model3.addConstr(1 + x3[2] + c3 == c2 + 3*x3[1], name="agent_2")

model3.write("ge_withtolls.lp")
model3.optimize()


#%%
print("General Equilibrium with tolls: \n")
print_res(model3, x3, p3)
print(f"Toll 1 is {round(c1, 3)}\nToll 2 is {round(c2, 3)}\nToll 3 is {round(c3, 3)}")
error = 1e-2

for i in range(3):
    assert np.abs(x[i].X - x3[i].X) <= error, f"Tolls did not work for flow {i}"

#%%

model4 = Model()

#Set Variables
x4 = model4.addVars(3, vtype="C", lb = 0.0, name = "edge")
p4 = model4.addVars(4, vtype="C", lb = 0.0, name ="path")
c = model4.addVars(3, vtype="C", name="tolls")

#Set Objective
model4.setObjective(x4[0] * (x4[0] + 1 + c[0]) + 3 * (x4[1]**2 + c[1]) + x4[2] * (x4[2] + 1 + c[2]), GRB.MINIMIZE)

model4.addConstr(p4[0] + p4[1] == 1, name="top_flow")
model4.addConstr(p4[2] + p4[3] == 1, name="botom_flow")
model4.addConstr(x4[0] == p4[0], name="top_edge")
model4.addConstr(x4[1] == p4[1] + p4[2], name="middle_edge")
model4.addConstr(x4[2] == p4[3], name="bottom_edge")
model4.addConstr(1 + x4[0] + c[0] == 3*x4[1] + c[1], name="agent_1")
model4.addConstr(1 + x4[2] + c[2] == 3*x4[1] + c[1], name="agent_2")

model4.Params.NonConvex = 2
model4.optimize()

print("General Equilibrium with tolls as part of costs: \n")
print_res(model4, x4, p4)
print(f"Toll 1 is {round(c[0].X, 3)}\nToll 2 is {round(c[1].X, 3)}\nToll 3 is {round(c[2].X, 3)}")

# %%
