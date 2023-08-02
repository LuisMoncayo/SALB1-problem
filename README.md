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
3. Optimise the value of the parameter by the SIMIO add-on OptQuest.
    - Optimisation using discreate IAT "OptQuest_Discreate_IAT.spfx"
    - Optimisation using stochastic IAT "OptQuest_Stochastic_IAT.spfx"

NOTE: User can download a free version of SIMIO from:

[SIMIO DOWNLOAD](https://www.simio.com/free-simulation-software/index.php)

This repository is related to the following paper:

Moncayo-Martínez, L.A.; Arias-Nava, E.H. Assessing by Simulation the Effect of Process Variability in the SALB-1 Problem. AppliedMath 2023, 3, 563-581. https://doi.org/10.3390/appliedmath3030030

[DOWNLOAD LINK](https://www.mdpi.com/2673-9909/3/3/30)

