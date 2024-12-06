# A Hierarchical Destroy and Repair Approach for Solving Very Large-Scale Travelling Salesman Problem

This repository contains the source code associated with the paper [A Hierarchical Destroy and Repair Approach for Solving Very Large-Scale Travelling Salesman Problem](https://arxiv.org/abs/2308.04639). For the instances used in the experiments and the best results, regardless of computation time, please refer to the following link: [Google Drive](https://drive.google.com/drive/folders/1maUAVQ27-PJ5Ekkq5X0xovZFTQkwPwQ6?usp=drive_link).

## Folder Descriptions

- **EAX**: Contains the implementation of the EAX algorithm. We made slight modifications to the original EAX code, which can be found at [EAX](https://github.com/nagata-yuichi/GA-EAX/tree/main/GA_EAX_1.0). For usage instructions, please refer to the [README.md](EAX/README.md) file in this folder.
- **LKH**: Contains the code for the LKH3 algorithm. The original code is available at [LKH3](http://akira.ruc.dk/~keld/research/LKH-3). For usage instructions, please see the [README.md](LKH/README.md) file in this folder.
- **HDR**: Contains the implementation of our HDR algorithm. For usage instructions, please refer to the [README.txt](HDR/README.md) file in this folder.

## Instance and Output

Input TSP instances must be in TSPLIB format, and output route files are also generated in TSPLIB format. For additional information, visit [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/). The instances used in our paper are from public test instances provided by the 8th DIMACS Implementation Challenge ([website](http://dimacs.rutgers.edu/archive/Challenges/TSP/download.html)). We used default parameters to generate the instances for our experiments.

## Requirements

To run the code, you need an environment that supports both C++ and Python.
