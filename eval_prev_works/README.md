# Evaluation of previous works.

We downloaded the code of several previous works that predict the MIC of peptides against bacterial targets. Those 
downloaded code were slightly modified in order to evaluate them on the QMAP 
benchmark. Each directory contains the code of a previous work. The code trains the model, then 
evaluates it on the QMAP benchmark for the five splits. The results are saved in the `results` folder inside each 
directory.

In addition, two baseline linear probing model are trained on the ESM650M embeddings, one for MIC and the other for 
HC50. They are implemented in the `Linear` and `HemoLinear` directories respectively.

The three notebooks generates the figures presented in the manuscript.

To rerun the experiments on the benchmark, simply run each main file in each directory. Make sure to activate the 
right uv environment for each previous works as they do not have the same python versions and dependencies.
