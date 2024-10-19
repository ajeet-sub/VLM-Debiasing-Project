# VLM-Debiasing-Project
## Instructions
### Accessing PACE and running compute node:
from terminal or using remote ssh extension on vscode:
    ssh login-ice.pace.gatech.edu

Inside home/your_folder
Open oddjobs interactive session: 
    run sinfo first to check which node is available and replace hugenod4
    srun -G 1 -w atl1-1-01-002-3-0 -t 0-07:00:00 -c 14 -n 1 --mem 60G --pty bash -i

    -G 1: number of GPUs to allocate
    -w hugenode4: particular node where the job will run # Q? does it always need to be hugenode4??
    -t 0-23:00:00: maximum wall-clock time for the job (23 hours)
    -c 14: 14 CPU cores
    -n 1: 1 task will run
    --mem 60G: 60 GB of memory
    --pty bash -i: This runs the job in an interactive session with a pseudo-terminal (pty) using an interactive Bash shell (bash -i), meaning you'll have a command-line shell where you can run commands interactively

https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/

### Github
 install Git pull extensions
 clone repo to a local folder
 configure ur username and passwor:
    git config user.name "Your Name"

## Installations
install conda:
    mkdir -p miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
    bash miniconda3/miniconda.sh -b -u -p miniconda3
    rm miniconda3/miniconda.sh

    miniconda3/bin/conda init bash


create conda environment --> conda create --name 
within conda environment
- install packages
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install -c conda-forge jupyterlab
- install jupyter extension on remote from VScode extensions
Try opening a jupyter notebook and choosing the kernel from the top right
    - select another kernel
    - python environments (and wait a bit)
If it doesn't show up
- try restarting VScode
-TRY to install ipy_kernel
    python3 -m ipykernel install --user OR python -m ipykernel install --user --name torch --display-name "Python (torch)"
    python3 -m pip install ipykernel
- try restarting VScode, o.w. I dunno :'D


tensorflow:
conda create --name tf python=3.10
conda activate tf
conda install -c conda-forge tensorflow=2.16.1

cuda support:
Driver Version: 555.42.06      CUDA Version: 12.5

slurm

The normal method to kill a Slurm job is:

    $ scancel <jobid>

You can find your jobid with the following command:

    $ squeue -u $USER


MYPORT=$(($(($RANDOM % 10000))+49152)); echo $MYPORT
-- 49349
 jupyter-notebook --no-browser --port=<MYPORT> --ip=0.0.0.0
srun -G 1 -w bignode5 -t 0-01:00:00 -c 14 -n 1 --mem 60G jupyter lab --no-browser --port=49349 --ip=0.0.0.0

To access the server, open this file in a browser:
        __________________
    Or copy and paste one of these URLs:
        __________________