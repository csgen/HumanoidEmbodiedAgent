# Monitor:
# note the current compute node name
hostname -f
# Show my quotas (Atlas helper - HOME/WORK/SCRATCH usage on atlas9)
hpc s
#Check cluster/queue load:
gstat
hpc q

#Inspect the queue limits (so you know what you can ask for)
# See all queues quickly
qstat -Q
# see all queues and ALL their parameters and resource details
qstat -Qf

# Show detailed limits for the serial queue
qstat -Qf serial | egrep 'resources_(default|max)|enabled|started'

# Show ACLs for a queue to see if it is restricted to some users
qstat -Qf largemem | egrep 'queue_type|enabled|started|resources_(default|max)|acl_'

#Request an interactive shell that fits the limits; some examples:
qsub -I -q serial -l select=1:ncpus=1:mem=4gb -l walltime=00:30:00
qsub -I -q parallel20 -l select=1:ncpus=4:mem=16gb -l walltime=01:00:00
# Or a GPU queue (change names/limits to your cluster’s)
qsub -I -q volta_gpu -l select=1:ncpus=10:ngpus=1:mem=40gb -l walltime=01:00:00

# Wait; when the prompt returns, you're on a compute node.
hostname -f   # note this FQDN, e.g. cnode-33-43-31.hpc.local

#Batch Jupyter (hands-off; you connect when it’s up)
#Create a PBS script that launches Jupyter on a compute node and prints the info required
mkdir $HOME/pbs
mkdir $HOME/pbs/logs
vim $HOME/pbs/sam_jlab_cpu.pbs
#    #!/bin/bash
#    #PBS -N sam_jlab_cpu
#    #PBS -q ood-cloud
#    # Keep asks small so it starts quickly; adjust if needed
#    #PBS -l select=1:ncpus=4:mem=8gb
#    #PBS -l walltime=04:00:00
#    # NOT RECOMMENDED - Write logs to a known place i.e. "/logs" in working dir
#    #defaults are jobname.o<jobid>/jobname.e<jobid> at $PBS_O_WORKDIR directly which is just as good
#    #PBS -o logs/hardcodedlogfilename.out
#    #PBS -e logs/hardcodedlogfilename.err
#    # “Join” streams: stderr is merged into stdout to get ONE log file; doing eo would merge stdout into stderr
#    #PBS -j oe
#
#    set -eoux pipefail
#    #-e → exit immediately on any non-zero status.
#    #-o pipefail → a pipeline fails if any command fails (not just the last).
#    #-u → treat unset variables as an error.
#    #-x → print each command as it executes (helps trace/debug).
#
#    cd "$PBS_O_WORKDIR"
#    #PBS sets PBS_O_WORKDIR to the directory where we store the script; helps keep its behaviour the same
#    echo "Host: $(hostname -f)"
#    echo "Start: $(date)"
#
#    # Make LD_LIBRARY_PATH safe under strict shells
#    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
#
#    # Load toolchain + conda hook + env
#    source /app1/ebenv
#    module avail /app1/ebapps/arches/flat-avx2/modules/lang/Anaconda3
#    module load Anaconda3/2023.09-0
#    source "$(conda info --base)/etc/profile.d/conda.sh"
#    conda activate /hpctmp/$USER/conda/envs/ai1008
#
#    which python; python -V
#    which jupyter; jupyter --version
#
#    # Pick a port and remember it
#    # Random port helps in Security-by-entropy (non obvious port; trip-wire) AND collision avoidance with other jobs
#    PORT=$(shuf -i 20000-60000 -n 1)
#    echo "Port: ${PORT}"
#
#    # Start Jupyter (token will be printed to stdout i.e. our .out log)
#    #jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT}
#    jupyter lab --no-browser --ip=127.0.0.1 --port=${PORT} \
#      --ServerApp.root_dir="${HOME}/projects"
#    #0.0.0.0 - listen on all network interfaces; Needed when something external on the node (e.g., Open OnDemand reverse proxy) must reach the HPC over the compute node’s network
#    #127.0.0.1 - listen only on loopback (localhost); more secure, no exposure on node's LAN; perfect for self SSH tunnelling

# start the job
JOBID=$(qsub jupyter_cpu.pbs)
JOBID=$(qsub $HOME/pbs/sam_jlab_cpu.pbs)

#check if job is running and its cnode details
qstat -ans $JOBID
#
#    venus01:
#                                                                Req'd  Req'd   Elap
#    Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
#    --------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
#    173410.venus01  <user_nm> ood-clo* jlab_cpu      --    1   4    8gb 04:00 R   --
#       cnode-33-43-31/0*4
#       Job run at Mon Sep 22 at 12:57 on (cnode-33-43-31:ncpus=4:mem=8388608kb)

# Show full historical info incl. exit status and log locations
qstat -xf $JOBID
qstat -xf $JOBID | egrep -i 'job_state|Exit_status|exec_host|Output_Path|Error_Path|comment'

#then, can view the error/output logs directly at the printed path:
cat /path/to/jobname.o<jobid>
cat /path/to/jobname.e<jobid>
tail -f ~/projects/quant-reasoning/sam_jlab_cpu.o173956
tail -f $HOME/pbs/logs/sam_jlab_cpu.o${JOBID%%.*} #extract just the number from the JOBID
#OR, use this command to view the error/output logs:
qcat -j $JOBID -t OU
qcat -j $JOBID -t OU -n 100
qcat -j $JOBID -t ER
#Use a refresh loop to monitor constantly
watch -n 2 'qcat -j <JOBID> -t OU | tail -n 50' #reprints every 2 seconds


#to terminate a job
qdel 173408.venus01
#to modify a queued job (eg Reduce memory & walltime) or move it to different queue
qalter -l select=1:ncpus=1:mem=2gb -l walltime=00:20:00 <JOBID>
qalter -q <newQueue> <JOBID>
qalter -o /new/path/to/logs $JOBID
# to increase current queue's walltime, pick a value within the queue’s max walltime
qalter -l walltime=02:00:00 <JOBID>
qstat -f <JOBID> | egrep 'Resource_List.walltime|Walltime'

# ϕ helpful commands:
# submit
qsub train_cpu.pbs
# list my jobs
qstat
# nodes & states
qstat -ans
# full details
qstat -f <jobid>
# Tail job output:
qcat -j <jobid> -t OU
# See processes on compute node:
qtop <jobid>
# Cancel if needed:
qdel <jobid>

#copy a file from HPC to Mac
scp username@HPCaddress:/path/to/remote/file /path/to/local/destination
#multiple files
scp username@HPCaddress:/path/to/remote/{file1,file2,file3} /path/to/local/destination
#If your Mac runs SSH and can be accessed from HPC:
scp /path/to/file mac_username@mac_ip:/path/to/destination


#tunnel from Mac OpenSSH (terminal) to that compute node via the login node (copy port from the output log first):
#single command using ProxyJump; then open http://localhost:<port> in browser
ssh -J <nusid>@atlas9.nus.edu.sg <nusid>@<compute-node-fqdn> -L <port>:localhost:<port>
# OR
ssh -L <port>:<compute-node-fqdn>:<port> <nusid>@atlas9.nus.edu.sg
#when prompted for token, paste whatever jupyter has printed in the output log...
#http://127.0.0.1:<port>/lab?token=<...ffb5ef46...>
# Alternatively, use fixed local port for a constant local experience with zero risk of on-node conflicts...
ssh -J <nusid>@atlas9.nus.edu.sg <nusid>@<compute-node-fqdn> -L 16108:localhost:$PORT
# Never need to reconfigure IDE/Browser since it always points to http://localhost:16108/?token=...

#monitor jupyter servers - run on the ssh into compute node since thats where the server is up
ss -ltnp | grep <PORT>
jupyter server list #needs enablement of the conda env first to make the command active
ps -u "$USER"

#sample CPU PBS Script:
#    #!/bin/bash
#    #PBS -N myjob
#    #PBS -l select=1:ncpus=8:mem=32gb
#    #PBS -l walltime=04:00:00
#    #PBS -j oe
#    cd $PBS_O_WORKDIR
#    source /path/to/miniconda/etc/profile.d/conda.sh
#    conda activate /hpctmp/$USER/envs/myenv
#    python train.py --config cfg.yaml

#sample GPU PBS Script:
#    #!/bin/bash
#    #PBS -N train_gpu
#    #PBS -q volta_gpu
#    #PBS -l select=1:ncpus=20:ngpus=2:mem=50gb
#    #PBS -l walltime=24:00:00
#    #PBS -j oe
#
#    cd "$PBS_O_WORKDIR"
#    source /app1/ebenv Python-3.10.4
#    source /hpctmp/$USER/conda/envs/myenv/bin/activate
#    python train.py --config cfg_gpu.yaml

#NUS Atlas includes helper commands and troubleshooting tips:
#	Quick submit (example shown in slides):
	qsub-auto -q serial "./a.out input1.dat" (creates _pbs_***.txt)
#Reference of what belongs where
#	•	Login nodes (atlas8/9): SSH, edit, submit jobs (qsub), check status (qstat, gstat, hpc q), transfer files. No compute.  ￼
#	•	Compute nodes: all CPU/GPU work (batch or interactive).
#	•	$HOME: code, configs, scripts, lockfiles. Snapshots exist.  ￼
#	•	$WORK: envs, active data, outputs (purge >60 days).  ￼
#	•	$SCRATCH: caches, temp & high-IO (purge >60 days).

#helpful Unix refreshers:
mv /source_folder/*jlab_cpu* /target_folder/

# ** OPTIONAL **
# choose IPYNB logs & token deterministically
TOKEN="${TOKEN:-$(python - <<'PY'
import secrets; print(secrets.token_urlsafe(24))
PY
)}"
# separate server log (besides the main outlog)
JLAB_LOG="/home/svu/<nusid>/pbs/logs/jupyter_${PBS_JOBID}.server.log"

# start Jupyter in background (bind to loopback; no retries)
jupyter lab \
  --no-browser \
  --ip=127.0.0.1 \
  --port="$PORT" \
  --ServerApp.port_retries=0 \
  --ServerApp.root_dir="$ROOT_DIR" \
  --ServerApp.token="$TOKEN" \
  >"$JLAB_LOG" 2>&1 & #2>&1 → “merge error and normal output.”
#It’s a very common trick when commands print useful info to stderr, this helps us grep/search or capture everything in one pipeline.

JPID=$!

# keep the job attached to the Jupyter process
wait "$JPID"
