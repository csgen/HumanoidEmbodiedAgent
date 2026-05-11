ssh <nusid>@<node>.nus.edu.sg
#$SCRATCH, $WORK, $HOME are various directories on the HPC system of which only $HOME survives 60-day housekeeping purges but also has a low 20GB capacity only

# Project scaffold (you already created this, keep as-is)
mkdir -p /$WORK/$USER/conda/envs /$WORK/$USER/data /$WORK/$USER/runs
mkdir -p /$SCRATCH/$USER/conda/pkgs /$SCRATCH/$USER/cache

# .condarc to keep envs/caches off $HOME
cat > $HOME/.condarc << 'EOF'
envs_dirs:
  - /$WORK/$USER/conda/envs
pkgs_dirs:
  - /$SCRATCH/$USER/conda/pkgs
EOF

# load the "ebenv" software environment, see https://bobcat.nus.edu.sg/hpc/support/ebenv for more information.
source /app1/ebenv

# load the required module to memory
module avail /app1/ebapps/arches/flat-avx2/modules/lang/Anaconda3
# load the Anaconda toolchain first
module load Anaconda3/2023.09-0 && module list
# load Conda's shell hook for this shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# confirm conda and python versions
conda --version
python --version
which conda
which python
which pip
#/app1/ebapps/arches/flat-avx2/software/Anaconda3/2023.09-0/bin/pip
ls -larth $HOME
cat $HOME/.condarc

# See if conda is picking up config files
conda config --show-sources
#    ==> /home/svu/e1554287/.condarc <==
#    envs_dirs:
#      - /hpctmp/$USER/conda/envs
#    pkgs_dirs:
#      - /scratch2/$USER/conda/pkgs
#
#    ==> cmd_line <==
#    debug: False
#    json: False

# check the effective values of above settings
conda config --show | egrep '^(envs_dirs|pkgs_dirs)'
conda info | egrep 'envs directories|package cache'
#    package cache : /scratch2/e1554287/conda/pkgs
#    envs directories : /hpctmp/e1554287/conda/envs

# list all envs
conda info --envs
#     conda environments:
#
#    base                     /app1/ebapps/arches/flat-avx2/software/Anaconda3/2023.09-0
#    ai1008                   /hpctmp/e1554287/conda/envs/ai1008

# since pointing is correct, can create using name
conda create -n ai1008
conda activate ai1008
# in PBS job scripts, use explicit paths
conda create -y -p /hpctmp/$USER/conda/envs/ai1008
conda activate /$WORK/$USER/conda/envs/ai1008

# Prefer conda-forge + strict priority for consistent solves
(ai1008) [ ]$ conda config --add channels conda-forge
(ai1008) [ ]$ conda config --set channel_priority strict
#-y = “yes to all prompts” (helpful for non-interactive environments like HPC)
(ai1008) [ ]$ conda install -y jupyterlab ipython ipykernel numpy pandas scipy scikit-learn matplotlib tqdm ipywidgets pyarrow plotly
(ai1008) [ ]$ conda install -y kaleido imageio
  306  conda install -y python-kaleido imageio
  307  conda install -y imageio-ffmpeg ffmpeg
  731  conda install -y scikit-plot
  760  conda install -y pydotplus
  768  conda install -y -c conda-forge mamba
  771  mamba install -c conda-forge pywavelets
  689  mamba install -y conda-forge::pytorch
  789  mamba install -y conda-forge::statsmodels
  943  mamba install -y conda-forge::seaborn
 1004  mamba install -y conda-forge::kaggle
 1010  mamba install -y conda-forge::kmodes
 1200  mamba install -y conda-forge::pykeen
(ai1008) [ ]$ plotly_get_chrome
mamba install -c conda-forge pywavelets
#The Chrome executable is now located at: /hpctmp/e1554287/conda/envs/ai1008/lib/python3.13/site-packages/choreographer/cli/browser_exe/chrome-linux64/chrome

# Make and register a named kernel (lives in $HOME; survives purges of $WORK and $SCRATCH; Jupyter sees this env by name)
python -m ipykernel install --user --name ai1008 --display-name "ai1008 (Atlas)"
#    Installed kernelspec ai1008 in /home/svu/e1554287/.local/share/jupyter/kernels/ai1008

#for tensorflow, python 3.12 is needed
(ai1008) [ ]$ mamba create -y -n tf312 -c conda-forge python=3.12 tensorflow keras jupyterlab ipython ipykernel numpy pandas scipy scikit-learn matplotlib tqdm ipywidgets pyarrow plotly
python -m ipykernel install --user --name tf312 --display-name "tf312 (Atlas - for TF)"
/hpctmp/e1554287/conda/envs/ai1008/bin/mamba install -n tf312 -c conda-forge pydot graphviz

#Data & env management (backup for reproducibility in case of purge)
conda env export --from-history > $HOME/projects/quant-reasoning/environment.yml




#to make native PYTORCH environment on CPU cluster (no conda):
source /app1/ebenv
module purge
python -V
#     Python 3.12.11
which python
#    /hpctmp/e1554287/conda/envs/tf312/bin/python
conda deactivate
python --version
#     Python 3.11.5
module avail 2>&1 | grep -i torch
#found PyTorch/2.1.2-foss-2023a to be most appropriate for non-GPU cluster
module load PyTorch/2.1.2-foss-2023a && module list
python --version
#    Python 3.11.3
which python
#    /app1/ebapps/arches/flat-avx2/software/Python/3.11.3-GCCcore-12.3.0/bin/python
pip install --upgrade pip setuptools wheel
pip install speechbrain pyroomacoustics soundfile ortools
/hpctmp/e1554287/conda/envs/ai1008/bin/mamba create -n ort311 -c conda-f^Cge ortools-python
ls /hpctmp/$USER
VENV=/hpctmp/$USER/.venvs/torch212
python -m venv "$VENV"
source "$VENV/bin/activate"

source /app1/ebenv
conda deactivate
python --version
which python
module purge
module avail 2>&1 | grep -i torch
module load PyTorch/2.1.2-foss-2023a && module list
python --version
which python
VENV=/hpctmp/$USER/.venvs/torch212
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel
pip install speechbrain pyroomacoustics soundfile ortools
 pip install --user speechbrain torchaudio torchvision pyroomacoustics ortools soundfile
python -c "import torch, torchaudio, torchvision, speechbrain, pyroomacoustics, ortools; print('✅ All imports OK')"
ll -arth /hpctmp/$USER/.venvs
pip list
python -m pip show torch torchaudio torchvision
python -m pip show speechbrain ortools
rm -rf /hpctmp/$USER/.venvs/torch212

#TRY AGAIN
module purge
python -V
#        Python 2.7.5
which python
#        /usr/bin/python
source /app1/ebenv

#        You are now using the "ebenv" software environment, see https://bobcat.nus.edu.sg/hpc/support/ebenv for more information.

module load PyTorch/1.12.0-foss-2022a
which python
#        /app1/ebapps/arches/flat-avx2/software/Python/3.10.4-GCCcore-11.3.0/bin/python
module load torchaudio/0.12.0-foss-2022a-PyTorch-1.12.0
module load torchvision/0.13.1-foss-2022a
python -m venv /hpctmp/$USER/.venvs/torch112
source /hpctmp/$USER/.venvs/torch112/bin/activate
(torch112) user@hpc:~$ which python
#        /hpctmp/e1554287/.venvs/torch112/bin/python
(torch112) user@hpc:~$ python -V
#        Python 3.10.4
(torch112) user@hpc:~$ pip install --upgrade pip setuptools wheel
python -m pip show torch torchaudio torchvision
# shows all 3

python -m pip show speechbrain ortools
#    WARNING: Package(s) not found: ortools, speechbrain
pip install speechbrain pyroomacoustics soundfile ortools
#gives sentencepiece error
pip install sentencepiece --prefer-binary speechbrain pyroomacoustics soundfile ortools
#works
(torch112) user@hpc:~$ python -c "import torch; import torchaudio; import torchvision; print('PyTorch ecosystem: OK')"
#    PyTorch ecosystem: OK
(torch112) user@hpc:~$ python -c "import speechbrain; print('SpeechBrain: OK')"
#    This version of torchaudio is old. SpeechBrain no longer tries using the torchaudio global backend mechanism in recipes, so if you encounter issues, update torchaudio to >=2.1.0.
#    SpeechBrain: OK
(torch112) user@hpc:~$ python -c "import pyroomacoustics; print('Pyroomacoustics: OK')"
#    Pyroomacoustics: OK
(torch112) user@hpc:~$ python -c "import soundfile; print('Soundfile: OK')"
#    Soundfile: OK
(torch112) user@hpc:~$ python -c "import ortools; print('OR-Tools: OK')"
#    OR-Tools: OK

#TO ADD THE KERNEL FOR IPY
# Install ipykernel
pip install ipykernel
python -m ipykernel install --user --name=torch12 --display-name="torch112 (Atlas - PyTorch)"
#install jupyter to start session directly on this venv
(torch112) user@hpc:~$ pip install jupyter jupyterlab
(torch112) user@hpc:~/projects/ideal-ML-train$ pip install hub matplotlib #hub has open source data like spoken MNIST


#make startup easy - we cd to /pbs/logs so running the JOB from the pbs/logs directory for logs etc is easier
cat > samstartup.sh << 'EOF'
#    > source /app1/ebenv
#    > module avail /app1/ebapps/arches/flat-avx2/modules/lang/Anaconda3
#    > module load Anaconda3/2023.09-0 && module list
#    > source "$(conda info --base)/etc/profile.d/conda.sh"
#    > conda activate ai1008
#    > cd $HOME/pbs/logs
#    > vim $HOME/pbs/sam_jlab_cpu.pbs
#    > EOF
EOF
source samstartup.sh

#some unix setups
vim ~/.vimrc
colorscheme desert

vim ~/.bashrc
# User specific aliases and functions
alias ls='ls --color=auto'
export PS1="\[\033[1;32m\]\u@\h:\[\033[1;34m\]\w\[\033[0m\]$ "
alias ll='ls -lhF'
alias gst='git status'
#some useful flags for ls
#    -l	Long format: shows permissions, owner, group, size, date, and filename
#    -a	All files: includes hidden files (those starting with .)
#    -r	Reverse order: reverses the sorting order
#    -t	Sort by time: sorts by last modification time (newest first, unless reversed)
#    -h	Human-readable: shows sizes as 1K, 234M, etc.
#    -F	Classify: adds / for directories, * for executables, @ for symlinks, etc.

#iPython theme
vim ~/.ipython/profile_default/ipython_config.json
#    {
#        "TerminalInteractiveShell": {
#            "colors": "linux",
#            "theme": {
#                "name": "monokai"
#            }
#        }
#    }

#Check IPython's documentation or use the Pygments styles list:
$ pygmentize -L styles

#git and sshkey setup
ssh-keygen -t ed25519 -C "s.soni87@yahoo.com"
#  Enter passphrase:

eval "$(ssh-agent -s)"
#  Agent pid 133714

ssh-add ~/.ssh/id_ed25519
#  Enter passphrase for /home/svu/e1554287/.ssh/id_ed25519:
#  Identity added: /home/svu/e1554287/.ssh/id_ed25519 s.soni87@yahoo.com

eval "$(ssh-agent -s)"
#  Agent pid 133775

chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
vim ~/.ssh/config
#Add the following"
#  Host github.com
#    HostName ssh.github.com
#    Port 443
#    User git
#    IdentityFile ~/.ssh/id_ed25519
#    IdentitiesOnly yes
#    ServerAliveInterval 60
#    ServerAliveCountMax 3
#    StrictHostKeyChecking yes
#    UserKnownHostsFile ~/.ssh/known_hosts

ssh -T git@github.com
#  Warning: Permanently added '[[ssh.github.com](http://ssh.github.com/)]:443,[20.205.243.160]:443' (ECDSA) to the list of known hosts.
#  Enter passphrase for key '/home/.ssh/id_ed25519':
#  Hi samarthsoni17! You've successfully authenticated, but GitHub does not provide shell access.

#check updated list of known hosts:
cat ~/.ssh/known_hosts

cd $HOME/projects
git clone git@github.com:samarthsoni17/ideal-ML-train.git

git config --global user.name "Samarth Soni"
git config --global user.email "s.soni87@yahoo.com"
git config --global push.default simple
git config --global color.ui auto
#to go beyond git default colour scheme:
git config --global color.status true
# OPTIONAL Set custom colors for status output:
git config --global color.status.added "green bold"
git config --global color.status.changed "yellow bold"
git config --global color.status.untracked "red bold"
git config --global color.diff.meta "magenta bold"
git config --global color.branch.current "yellow reverse"
#to see current color settings:
git config --get-regexp color

#only show files changes in a commit:
git show --name-only <hash>

If you later confirm you need GPU TF on a specific queue/node, tell me the GPU model and driver/CUDA shown there, and I’ll give you the exact install line.

#to rename all files in a folder with a string appended at the front:
$ bash << 'EOF'
> for f in *; do
>   if [[ -f "$f" ]]; then
>     echo executing- mv "$f" "run1$f"
>     mv "$f" "run1_$f"
>   fi
> done
> EOF

#to rename all files in a folder with a string appended at the front but ignore files that already have that string:
$ bash << 'EOF'
for f in *; do
  if [[ -f "$f" && "$f" != run1_* ]]; then
    echo executing mv "$f" "run2_$f"
    mv "$f" "run2_$f"
  fi
done
EOF

#to rename all files back to original names:
$ bash << 'EOF'
for f in run2_*; do
  if [[ -f "$f" ]]; then
    newname="${f#run2_}"
    echo renaming "$f" back to "$newname"
    mv "$f" "$newname"
  fi
done
EOF

#to use kaggle API to download data directly:
mkdir ~/.kaggle
vim ~/.kaggle/kaggle.json #then paste in kaggle API token json content
!chmod 600 ~/.kaggle/kaggle.json