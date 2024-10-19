# VLM-Debiasing-Project
## Instructions
### Accessing PACE and running compute node:
from terminal or using remote ssh extension on vscode:
    ssh login-ice.pace.gatech.edu

Inside home/your_folder
Open oddjobs interactive session: 
1. run `sinfo` first to check which node is available and replace hugenod4
2. `srun -G 1 -w hugenod4 -t 0-07:00:00 -c 14 -n 1 --mem 60G --pty bash -i`
```
    -G 1: number of GPUs to allocate
    -w hugenode4: particular node where the job will run # Q? does it always need to be hugenode4??
    -t 0-23:00:00: maximum wall-clock time for the job (23 hours)
    -c 14: 14 CPU cores
    -n 1: 1 task will run
    --mem 60G: 60 GB of memory
    --pty bash -i: This runs the job in an interactive session with a pseudo-terminal (pty) using an interactive Bash shell (bash -i), meaning you'll have a command-line shell where you can run commands interactively
```
3. The normal method to kill a Slurm job is:

    $ scancel jobid

4. You can find your jobid with the following command:

    $ squeue -u $USER

https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/

### Github
1. install Git pull extensions
2. clone repo to a local folder using the terminal
3. configure ur username and email:
    git config user.name "Your Name"
    git config user.email "email"

## Installations
### conda
inside **scratch folder**:
```
mkdir -p miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
bash miniconda3/miniconda.sh -b -u -p miniconda3
rm miniconda3/miniconda.sh
miniconda3/bin/conda init bash
```
Restart the terminal

### Packages Installation
-------------------------------------------------------------------

1. create conda environment --> 
`conda create --name "vlm-debiasing" `
2. within conda environment
3. install packages
    - PyTorch
    `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
    check installation:
        ```shell
        python
        import torch
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        ```
    - Jupyter `conda install -c conda-forge jupyterlab`
        - install jupyter extension on remote from VScode extensions
        - Try opening a jupyter notebook and choosing the kernel from the top right
            - select another kernel
            - python environments (and wait a bit)
            - If it doesn't show up
                - try restarting VScode
                - TRY to install ipy_kernel
                `python3 -m ipykernel install --user OR python -m ipykernel install --user --name torch --display-name "Python (torch)"` or
                `python3 -m pip install ipykernel`
            - try restarting VScode, o.w. I dunno :'D
        - to make jupyter see a compute node instead of the login node:
            - open a new terminal
            - activate the conda environment you want to run jupyter from
            - start a compute job:
            `srun -G 1 -w bignode5 -t 0-01:00:00 -c 14 -n 1 --mem 60G jupyter lab --no-browser --port=49349 --ip=0.0.0.0`
            - copy the 2nd link to the jupyter server that shows up at the end of the terminal
                ```shell
                To access the server, open this file in a browser:
                    1__________________
                Or copy and paste one of these URLs:
                    2__________________
                    3__________________
                ```
            - go back to ur jupyter notebook and choose the kernel from the top right then
                - select another kernel
                - an existing jupyter server
                - paste the link you copied
            - check that your jupyter notebook is seeing the computing node by running the following code inside ur jupyter notebook
                ```python
                import os

                def is_slurm_job():
                    # Check if common SLURM environment variables are set
                    return 'SLURM_JOB_ID' in os.environ or 'SLURM_TASKS_PER_NODE' in os.environ

                # Check if the Jupyter notebook is running under a SLURM job
                if is_slurm_job():
                    print("Running on a SLURM-managed cluster.")
                    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
                    print(f"SLURM_NODELIST: {os.environ.get('SLURM_NODELIST')}")
                else:
                    print("Running locally (not on a SLURM-managed cluster).")
                ```
    - Pandas `conda install pandas`
    - tqdm `pip install tqdm`
    - matplotlib `conda install matplotlib`

## Data download to remote server
inside your scratch folder:
- `mkdir -p e-daic`
- `wget -r -np -nH --cut-dirs=1 -P ./e-daic http://example.com/path/to/files/`

## To untar the e-daic dataset
run `python ./scripts/untar_data.py --root-dir scratch/vlm-debiasing/e-daic_orig_data --dest-dir scratch/vlm-debiasing/e-daic_untar_data` but change destination and root directories
