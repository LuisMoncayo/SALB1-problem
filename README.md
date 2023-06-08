# Solving the SALB-1 problem in process variability.
This project has the files of a proposed three-part algorithm to find a solution to a stochastic Simple Assembly Line Balancing (SALB-1) problem. In this case the SALB-1 problem is solve; thus, the number of open cells is unknown as well as the assignation of the task to the open cells.

The three parts are:
1. Solve the mixed-integer linear programming model using GLPK via Pyomo. The files are:
    - pyomo_implementation.py
    - gearbox_instance.xlsx
2. Create a SIMIO simulation model add variability to the balancing using a set of parameter. 
    - The file "Scaled_Model.spfx" has the following experiments:
        - Verification (model verification)
        - Workers_Speed (speed of the workers in cells)
        - IAT (inter--arrival time)
        - Nu_Workers (number of workers in cells)
4. Optimise the value of the parameter by the SIMIO add-on OptQuest.

Open the corresponding file to open the files using to accomplish each part.
