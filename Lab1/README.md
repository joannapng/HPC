# Lab 1: Code optimizations on Sobel Filter
##### Ioanna-Maria Panagou, 2962 & Nikolaos Chatzivangelis, 2881
---------------------------------------

## Contents
- **makefile**
- **O0** - Each subfolder in folder **O0** contains the source file **sobel.c** for the respective configuration
- **fast** - Each subfolder in folder **fast** contains the source file **sobel.c** for the respective configuration
- **times.sh** - bash script that automatically compiles and runs each configuration 25 times for both -O0 and -fast and stores the results in _times.csv_ and _times_fast.csv_
- **HPC_Lab1.pdf** - Contains a short description for each configuration and our experimental results
- **HPC_Lab1.xlsx** - Spreadsheet that contains the result of our runs. If an optimization is highlighted in red instead of green, then it is not included in the next configuration.

## How to use **times.sh**
- Make sure that _input.grey_ and  _golden.grey_ is in the **Lab1** folder and that **Lab1** also contains folder **O0** and **fast**.
- Run
    ./times.sh
    
## Run each configuration separately
- Copy the _input.grey_ and _golden.grey_ files inside the subfolder that contains the source file of the configuration you wish to run.
- Make sure that for O0 line 11 in the makefile is commented.
- For fast, uncomment line 11 and comment 12
- Run
    make
    ./sobel

