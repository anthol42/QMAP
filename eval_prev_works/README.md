# Evaluation of previous works.

We downloaded the code of several previous works that predict the MIC of peptides against bacterial targets or predict 
the hemolytic potential of AMPs. Those downloaded code were slightly modified in order to evaluate them on the QMAP 
benchmark. Each directory contains the code of a previous work. The code trains the model, then 
evaluates it on the QMAP benchmark for the five splits. The results are saved in the `results` folder inside each 
directory.

The three notebooks generate the figures presented in the manuscript.

To rerun the experiments on the benchmark, simply run each main file in in each directory. Make sure to activate the 
right uv environment for each previous works as they do not have the smame python versions and dependencies.
