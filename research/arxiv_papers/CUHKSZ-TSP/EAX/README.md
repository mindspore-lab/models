# Introduction

This folder contains the source code for EAX, which was used in our experiments. The original source code is available from [EAX](https://github.com/nagata-yuichi/GA-EAX/tree/main/GA_EAX_1.0). We made slight adjustments to the original code.

# Folder Functions

- **Detail**: Stores the final results of each execution.
- **Error**: Contains error files from each execution.
- **Exe_cpp**: Contains the EAX code.
- **Ins**: Contains instances being tested. Due to the large size of benchmark instances, please download them from TSP websites and place them here. Refer to E10K.0 for the file format.
- **Output**: Stores part results from each execution.
- **Sol**: Stores part results from each execution.
- **Summary_new**: Contains the summary of results.

# File Functions

- **compile.py**: Compiles the code and generates a list of tasks. You can modify related settings in 'LaRen.sh'.
- **fn2.txt**: Lists instances. You can adjust it as needed.
- **LaRen.sh**: Compilation script. The option "-l 10" specifies the number of executions for each instance.
- **Qsub.sh**: Submission script. The option "-c 10" specifies the number of threads for running tasks.
- **submit_multiprocess.py**: Submission script. You can modify related settings in 'Qsub.sh'.
- **subtsk2.sh**: List of the executions.
- **syngen_new.py**: Generates the summary file.

# Execution Steps

Step 1: Put the instances being tested in the "Ins" folder. Then, compile the code and generate the list of tasks by running:

```bash
./LaRen.sh
```

Step 2: Execute the tasks and wait for completion with:

```bash
./Qsub.sh
```

Step 3: Once the executions are finished, summarize the results. The final summary file can be found in 'summary_new'. The columns in the file represent: instance, best-known, best, average, preprocessing time, total time, the gap between best and best-known, and the gap between average and best-known, respectively.

Generate the summary file by running:

```bash
python2 syngen_new.py
```
