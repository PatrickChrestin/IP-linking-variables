from gurobipy import *

def solve(N, M, k, coef_matrix, r_sum, visualization = False):
    # sort variables and constraints for visualization
    sorted_vars = {}
    sorted_constraints = {}

    blocks = range(1,k+1)

    # Modell initialisieren
    model = Model("staircasedetection")
    model.modelSense = GRB.MINIMIZE

    # Variablen initialisieren
    x,y,z = {}, {}, {}
    for b in blocks:
       for j in N:
           x[j,b] = model.addVar(name=f"x_{j}_{b}", vtype=GRB.BINARY)
       for i in M:
           y[i,b] = model.addVar(name=f"y_{i}_{b}", vtype=GRB.BINARY)
    for j in N:
        z[j] = model.addVar(name=f"z_{j}", vtype=GRB.BINARY)

    model.setObjective(quicksum(z[j] for j in N))
    model.update()

    # var i in genau einem Block oder in zwei, wenn linking (z_i = 1)
    for j in N:
        for b in range(1, k):
            model.addConstr(x[j,b] + x[j,b+1] <= 1 + z[j])
    
    # var i taucht nur in aufeinanderfolgenden Blöcken auf
    for j in N:
        for b in range(1, k-1):
            for l in range(2, k+1-b):
                model.addConstr(x[j,b] + x[j,b+l] <= 1)

    # jede constraint ist genau einem Block zugeordnet
    for i in M:
        model.addConstr(quicksum(y[i,b] for b in blocks) == 1)
    
    # jeder Block hat mindestens eine constraint
    for b in blocks:
        model.addConstr(quicksum(y[i,b] for i in M) >= 1)

    # für jede constraint existieren die vars in dem zugewiesenen Block
    for b in blocks:
        for i in M:
            model.addConstr(quicksum((coef_matrix[i-1,j-1]*x[j,b]) for j in N) >= r_sum[i-1]*y[i,b])

    model.optimize()

    runtime = model.Runtime
    opt = -1
    # Ausgabe der Loesung.
    if model.status == GRB.OPTIMAL:
        runtime = model.Runtime
        opt = model.ObjVal
        if visualization:
            # initialize dicts
            for i in range(1,k):
                sorted_vars[str(i)] = []
                sorted_vars[str(i)+","+str(i+1)] = []
                sorted_constraints[str(i)] = []
            sorted_vars[str(k)] = []
            sorted_constraints[str(k)] = []

            for b in blocks:
                for j in N:
                    # if linking and in block
                    if (z[j].x > 0 and x[j,b].x > 0):
                        if(b<k and x[j,b+1].x > 0): # prohibits linking variable entry in last block and assignment of linking vars to multiple linking blocks
                            sorted_vars[str(b)+","+str(b+1)].append(j)
                    # else if in block
                    elif (x[j,b].x > 0):
                        sorted_vars[str(b)].append(j) 
                for i in M:
                    if (y[i,b].x > 0):
                        sorted_constraints[str(b)].append(i)

        # print(f"\nOptimaler Zielfunktionswert: {model.ObjVal:.2f}\n")
        # for j in N:
        #     for b in blocks:
        #         if (x[j,b].x > 0):
        #             print(f"Variable {j} ist im Block {b} enthalten.")
        #     if (z[j].x > 0):
        #         print(f"Variable {j} ist linking.")
        # for i in M:
        #     for b in blocks:
        #         if (y[i,b].x > 0):
        #             print(f"Row {i} ist im Block {b} enthalten.")
    else:
        print(f"Keine Optimalloesung gefunden. Status {model.status}")
        
    return model, opt, sorted_vars, sorted_constraints, runtime