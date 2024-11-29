# Introduction

This folder contains the source code for our algorithm, the Hierarchical Destroy-and-Repair (HDR) approach. For more details, please refer to the paper [A Hierarchical Destroy and Repair Approach for Solving Very Large-Scale Travelling Salesman Problem](https://arxiv.org/abs/2308.04639).

# Folder Functions

- **Src**: Contains the source code.

# File Functions

- **compile.sh**: Compiles the code and generates the executable file "SOLVER".

# Execution Steps

Step 1: Compile the code and generate the executable file "SOLVER" by running:

```bash
./compile.sh
```

Step 2: Run the instance you want with:

```bash
./SOLVER.sh -input A -output B --seed C -time D -sol E -ope F -edge G
```

- **-input**: Mandatory. A should be the path of the problem file in TSPLIB format.
- **-output**: Mandatory. B should be the path of the output file where the best tour will be recorded.
- **--seed**: Optional. Controls the initial seed, with a default value of 1.
- **-time**: Optional. Specifies the total run time. If not provided, the program will not terminate based on time.
- **-sol**: Optional. Sets the number of solutions to be generated, defaulting to 10.
- **-ope**: Optional. Controls the number of destroy and repair operations in each hierarchy. The number is calculated as n/F, where n is the number of cities in each hierarchy. The default value is 90.
- **-edge**: Optional. Specifies the number of edges destroyed during each destroy and repair operation, with a default value of 500.

# Experiments 1 and 2 for the Paper

For each instance, the algorithm is executed 10 times with seed values ranging from 1 to 10. Set D to n/5 (where n is the size of the problem) if n is less than 1,000,000; otherwise, set D to 100,000. Other parameters use their default values.

Therefore, the command is:

```bash
./SOLVER -input A -output B --seed C -time n/5
```

or:

```bash
./SOLVER -input A -output B --seed C -time 100,000
```
