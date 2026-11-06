# levy-processes-assignment
Group assignment for Levy Processes course in the MSc in Mathematical Finance at ISEG Lisbon

## Structure
Files in the `scripts` directory will call the functions from our "library" `src/vg_cir` and use the appropraite parameters and data. 

The files in `src/vg_cir` are written as re-usable, modular functions. These can also be used across different parts of the assignment. 

Part 1 saves the option data as a CSV. Parts 2 reads this CSV, estimates the model's parameters and saves those parameters to a file. Part 3 reads those parameters and simulates the paths. Part 4 takes the simulated paths and computes the UOBC option price. 

## Note
I put some comments at the top of each file with what I think should go in there. Of course this structure is just my idea and a guide. If there are parts you think should be different, or things need to be added/removed, feel free to do so. 

## Dev Update: 6/11/2026
Simulation code is up and running. To run the code follow the steps below:

## Set-up
From the repo root `levy-processes-assignment` type:
- `python -m pip install -e .`
- `python -m pip install pytest`

You can then run the files from the root folder and the imports should work. Also you can run the tests with:
`python -m pytest`