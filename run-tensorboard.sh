# Usage:
# bash run-tensorboard.sh <logdir> <port>

LOGDIR=$1
PORT=$2

nohup srun -c 2 -p gpu,mem --time 0-12 --job-name tensorboard-${PORT} bash -c "source .env/bin/activate; tensorboard --logdir ${LOGDIR} --port ${PORT} --samples_per_plugin scalars=1000,images=100" > log-tensorboard-${PORT}.txt 2>&1 &
