import os
import glob
from gym_pycr_pwcrack.agents.config.agent_config import AgentConfig
from gym_pycr_pwcrack.dao.experiment.client_config import ClientConfig
from gym_pycr_pwcrack.dao.agent.agent_type import AgentType
from gym_pycr_pwcrack.util.experiments_util import util
from gym_pycr_pwcrack.util.experiments_util import plotting_util
from gym_pycr_pwcrack.dao.network.cluster_config import ClusterConfig
from gym_pycr_pwcrack.dao.experiment.runner_mode import RunnerMode

def default_config() -> ClientConfig:
    """
    :return: Default configuration for the experiment
    """
    agent_config = AgentConfig(gamma=0.99, alpha=0.0005, epsilon=1, render=False, eval_sleep=0.0,
                                                eval_episodes=10, train_log_frequency=1000,
                                                video=False, eval_log_frequency=1,
                                                video_fps=5, video_dir=util.default_output_dir() + "/results/videos",
                                                num_iterations=1000000,
                                                eval_render=False, gifs=True,
                                                gif_dir=util.default_output_dir() + "/results/gifs",
                                                eval_frequency=100, video_frequency=10,
                                                save_dir=util.default_output_dir() + "/results/data",
                                                checkpoint_freq=100, input_dim=6 * 30,
                                                output_dim=114,
                                                shared_hidden_layers=2, shared_hidden_dim=128,
                                                batch_size=32,
                                                gpu=False, tensorboard=True,
                                                tensorboard_dir=util.default_output_dir() + "/results/tensorboard",
                                                optimizer="Adam", lr_exp_decay=False, lr_decay_rate=0.999,
                                                gpu_id=0,
                                                lr_progress_decay=False, lr_progress_power_decay=4,
                                                max_gradient_norm=0.5,
                                                render_steps=20, illegal_action_logit=-100, buffer_size=1000000,
                                                tau = 1.0, learning_starts = 50000, train_freq=4, gradient_steps=1,
                                                target_update_interval=10000, exploration_fraction=0.99,
                                                exploration_initial_eps=1.0, exploration_final_eps=0.05
                                                )
    env_name = "pycr-pwcrack-simple-cluster-v1"
    cluster_config = ClusterConfig(agent_ip="172.18.1.191", agent_username="agent", agent_pw="agent",
                                   server_connection=False)
    client_config = ClientConfig(env_name=env_name, agent_config=agent_config,
                                 agent_type=AgentType.DQN_BASELINE.value,
                                 output_dir=util.default_output_dir(),
                                 title="DQN-Baseline v1",
                                 run_many=False, random_seeds=[0, 999, 299, 399, 499],
                                 random_seed=399, cluster_config=cluster_config, mode=RunnerMode.TRAIN_ATTACKER.value)
    return client_config


def write_default_config(path:str = None) -> None:
    """
    Writes the default configuration to a json file

    :param path: the path to write the configuration to
    :return: None
    """
    if path is None:
        path = util.default_config_path()
    config = default_config()
    util.write_config_file(config, path)


# Program entrypoint
if __name__ == '__main__':

    # Setup
    args = util.parse_args(util.default_config_path())
    experiment_title = "DQN simple v1 cluster"
    if args.configpath is not None and not args.noconfig:
        if not os.path.exists(args.configpath):
            write_default_config()
        config = util.read_config(args.configpath)
    else:
        config = default_config()

    # Plot
    if args.plotonly:
        if args.csvfile is not None:
            plotting_util.plot_csv_files([args.csvfile],
                                        config.output_dir + "/results/plots/" + str(config.random_seed) + "/")
        elif config.run_many:
            csv_files = []
            for seed in config.random_seeds:
                p = glob.glob(config.output_dir + "/results/data/" + str(seed) + "/*_train.csv")[0]
                csv_files.append(p)
            plotting_util.plot_csv_files(csv_files, config.output_dir + "/results/plots/")
        else:
            p = glob.glob(config.output_dir + "/results/data/" + str(config.random_seed) + "/*_train.csv")[0]
            plotting_util.plot_csv_files([p], config.output_dir + "/results/plots/" + str(config.random_seed) + "/")

    # Run experiment
    else:
        if not config.run_many:
            train_csv_path, eval_csv_path = util.run_experiment(config, config.random_seed)
            if train_csv_path is not None and not train_csv_path == "":
                plotting_util.plot_csv_files([train_csv_path], config.output_dir + "/results/plots/"
                                             + str(config.random_seed)+ "/")
        else:
            train_csv_paths = []
            eval_csv_paths = []
            for seed in config.random_seeds:
                if args.configpath is not None and not args.noconfig:
                    if not os.path.exists(args.configpath):
                        write_default_config()
                    config = util.read_config(args.configpath)
                else:
                    config = default_config()
                train_csv_path, eval_csv_path = util.run_experiment(config, seed)
                train_csv_paths.append(train_csv_path)
                eval_csv_paths.append(eval_csv_path)

            plotting_util.plot_csv_files(train_csv_paths, config.output_dir + "/results/plots/")
