# Introduction 

This folder contains the source code for LKH3, which we used in our experiments. The original source code is available from [LKH3](http://akira.ruc.dk/~keld/research/LKH-3).

# Folder Functions

- **DOC**: Contains papers related to LKH.

- **SRC**: Stores the code.

# File Functions

- **C10k.0_1.par**: Demo .par file.

- **Makefile**: Compiles the code and generates the executable file "LKH".

# Execution Steps

Step 1: Compile the code and generate the executable file "SOLVER" by running:

```make```

Step 2: Write a .par file 

The following format is specific to our experiment:

```
PROBLEM_FILE = A
CANDIDATE_SET_TYPE = POPMUSIC
INITIAL_PERIOD = 1000
TIME_LIMIT = B
RUNS = 1
SEED = C
TOUR_FILE = D
```

- A is the path of the problem file.
- B sets the total run time.
- C sets the initial seed.
- D specifies the path for the file where the best tour will be saved.

Step 3: Run the desired instance with:

```./LKH xxx.par```

Where `xxx.par` is the .par file you want to execute.

# Our Experiment

For each instance, we run 10 times with SEED values from 1 to 10. Set TIME_LIMIT to n/5 (where n is the size of the problem) if n is less than 1,000,000; otherwise, set TIME_LIMIT to 100,000. Other parameters follow the demo settings.

