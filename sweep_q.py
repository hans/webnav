import os.path
import subprocess
import sys
import uuid

import numpy as np


ARGS = {
    "learning_rate": (0.0001, 0.01),
    "epsilon": (0.05, 0.3),

    "oracle_freq": (0.05, 0.4),
    "goal_reward": (2.0, 10.0),
    "match_reward": (0.25, 1.0),
}


def sample(args):
    return {key: np.random.uniform(low, high)
            for key, (low, high) in args.iteritems()}


def run_script(args):
    exp_id = str(uuid.uuid4())
    args["logdir"] = os.path.join("/local_home", "jon", "q_%s" % args["task_type"], exp_id)
    os.mkdir(args["logdir"])

    if args["task_type"] == "navigation":
        args["oracle_freq"] = 0.0

    command = (
            """CUDA_VISIBLE_DEVICES={gpu} python webnav/scripts/train_qlearning_comm.py \\
                    --data_type wikispeedia \\
                    --wiki_path data/wikispeedia/wikispeedia.pkl \\
                    --emb_path data/wikispeedia/wikispeedia_embeddings.npz \\
                    --task_type {task_type} --batch_size 256 --n_epochs 200 \\
                    --beam_size 32 --path_length 10 --eval_interval 100 \\
                    --summary_interval 20 --logdir {logdir} \\
                    --learning_rate {learning_rate} --epsilon {epsilon} \\
                    --oracle_freq {oracle_freq} \\
                    --goal_reward {goal_reward} --match_reward {match_reward} \\
                    > {logdir}/out 2>&1""")
    command = command.format(**args)
    print command

    subprocess.call(command, shell=True)


if __name__ == "__main__":
    gpu_id = sys.argv[1]
    task_type = sys.argv[2]
    while True:
        sample_args = sample(ARGS)
        sample_args["gpu"] = gpu_id
        sample_args["task_type"] = task_type
        run_script(sample_args)
