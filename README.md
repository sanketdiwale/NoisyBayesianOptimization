# NoisyBayesianOptimization

Working tested with python version 3.8.5
## Install dependencies:
```
pip install -r requirements
```
Install the module AAD by running
```
python setup.py install
```

## Run the Bayesian optimization
```
cd AAD/Strategies/1layerqGPOpt
```
Run the bayesian optimization code
```
python BoostedGPLCB.py 1
```
The script takes as input a integer batch number and records the results to a directory with the batch number appended to separate results from different batch runs.

If running with a computer cluster running SLURM a batch of simulations can be run using
```
sbatch run_script.sh
```