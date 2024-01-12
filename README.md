*NOTE*: This repo is used for reviewing

## How to replicate the experiments
NOTE that in order to anonymize the code, we have removed usernames of wandb and cluster project names. The code may not be runnable immediately. But it should be easy to fix by using your own usernames. 
### 1. Config your Python environment
```bash
conda env create -f environment.yml
```

### 2. Run the script
```bash
cd scripts/
./run_mpe_cpu.sh

```
### 3. Check your wandb plots.

### 4. Run the scripts on the other two benchmarks, SMACv2 and XOR game.
