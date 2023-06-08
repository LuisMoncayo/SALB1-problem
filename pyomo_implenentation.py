#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:03:01 2023

@author: luismoncayo
"""
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import math
import networkx as nx
import matplotlib.pyplot as plt
import pyomo.environ as pyo

######################################
## IMPORT DATA
######################################
inst_data = pd.DataFrame()
inst_data = pd.read_excel(r'Data/gearbox_instance.xlsx', sheet_name='TwoStageGearbox')

task = inst_data['Task'].tolist()
precedence = inst_data['Precedence'].tolist()
processing_times = inst_data['Time'].tolist()

# set the edges (from, to)
precedence_relationships = []
for p in range(len(precedence)):
    if type(precedence[p]) == str:
        i = task[p]
        index = precedence[p].split(",")
        for j in range(len(index)):
            precedence_relationships.append([int(index[j]),i])
    elif math.isnan(precedence[p]):
        pass
    else:
        i = precedence[p]
        j = task[p]
        precedence_relationships.append([i,j])

# attach the process time to the tasks
process_time = dict()
for key, value in zip (task, processing_times) :
    process_time[key] = value

######################################
## CREATE THE DI-GRAPH
######################################
G = nx.DiGraph()

G.add_edges_from((precedence_relationships))
list_edges = list(G.edges)
list_nodes = list(G.nodes)

#plt.figure(figsize=(7.5, 7.5))
#nx.draw_networkx(G)
#plt.show()

##############################################
## IMPLEMENTATION IN PYOMO                  ##
## SALB 1: Get the minimum number of cells, ##
##         given a cycle time               ##
##############################################
model = pyo.ConcreteModel()

nu_cells = 4
cycle_time = 70
list_cells = list(range(1,nu_cells+1,1))

model.N = pyo.Set(initialize=list_nodes) # list of tasks (i.e. nodes); i=1,..,N
model.M = pyo.Set(initialize=list_cells) # list of cells; j=1,...,M
model.x = pyo.Var(model.N,model.M, domain=pyo.Binary)
model.y = pyo.Var(model.M, domain=pyo.Binary)

def objective_SALB1(model):
    return sum(model.y[i] for i in model.M)
model.obj = pyo.Objective(rule=objective_SALB1, sense=pyo.minimize)

def cycle_time_cell(model, j):
    return sum(process_time[i]*model.x[i,j] for i in model.N) <= cycle_time*model.y[j]
model.cycle_cell = pyo.Constraint(model.M, rule=cycle_time_cell)

def task_assignation(model, i):
    return sum(model.x[i,j] for j in model.M) == 1
model.task_assig = pyo.Constraint(model.N, rule=task_assignation)

model.precedence_task = pyo.ConstraintList()
for edge in precedence_relationships:
    from_task = edge[0]
    to_task = edge[1]
    model.precedence_task.add(expr=sum(j*model.x[from_task,j] for j in model.M) <=  sum(j*model.x[to_task,j] for j in model.M))

model.cells = pyo.ConstraintList()
for j in range(nu_cells-1):
        model.cells.add(expr=model.y[j+2] <= model.y[j+1])
#model.pprint()

results = pyo.SolverFactory('glpk').solve(model)

print("The minimum number of cells is ", pyo.value(model.obj) )

for j in range(nu_cells):
    cell_time = 0
    print()
    print("Cell",j+1, "produces the tasks: ", end="")
    for n in list_nodes: 
        if(pyo.value(model.x[n,j+1]) == 1.0):
            cell_time += process_time[n]
            print(n, end=", ")
    print()
    print("The time content in cell",j+1," is",  cell_time, end="")
    print()
    print("Efficiency of cell",j+1," is",round((cell_time/cycle_time)*100,2),"%", end="")
    print()


##############################################
## IMPLEMENTATION IN PYOMO                  ##
## SALB 2: Get the minimum cycle time,      ##
##         given a number of cells          ##
##############################################
model = pyo.ConcreteModel()

nu_cells = 3
list_cells = list(range(1,nu_cells+1,1))

model.N = pyo.Set(initialize=list_nodes) # list of tasks (i.e. nodes); i=1,..,N
model.M = pyo.Set(initialize=list_cells) # list of cells; j=1,...,M
model.x = pyo.Var(model.N,model.M, domain=pyo.Binary)
model.C = pyo.Var(domain=pyo.PositiveReals)

model.objective = pyo.Objective(expr=model.C)

def cycle_time_cell(model, j):
    return sum(process_time[i]*model.x[i,j] for i in model.N) <= model.C
model.cycle_cell = pyo.Constraint(model.M, rule=cycle_time_cell)

def task_assignation(model, i):
    return sum(model.x[i,j] for j in model.M) == 1
model.task_assig = pyo.Constraint(model.N, rule=task_assignation)

model.precedence_task = pyo.ConstraintList()
for edge in precedence_relationships:
    from_task = edge[0]
    to_task = edge[1]
    model.precedence_task.add(expr=sum(j*model.x[from_task,j] for j in model.M) <=  sum(j*model.x[to_task,j] for j in model.M))
#model.pprint()

results = pyo.SolverFactory('glpk').solve(model)
print()
print()
print("The minimum cycle time is ", pyo.value(model.objective) )

for j in range(nu_cells):
    cell_time = 0
    print()
    print("Cell",j+1, "produces the tasks: ", end="")
    for n in list_nodes: 
        if(pyo.value(model.x[n,j+1]) == 1.0):
            cell_time += process_time[n]
            print(n, end=", ")
    print()
    print("The time content in cell",j+1," is",  cell_time, end="")
    print()
    print("Efficiency of cell",j+1," is",round((cell_time/pyo.value(model.objective))*100,2),"%", end="")
    print()  
