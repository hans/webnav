from rllab.sampler.base import BaseSampler


class VecEnvExecutor(object):
    def __init__(self, env, n, max_path_length):
        self.env = env
        self.n = n
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self.ts = np.zeros((n,), dtype='int')
        self.max_path_length = max_path_length

    def step(self, actions):
        observations, dones, rewards = env.step_batch(actions)
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        dones[self.ts >= self.max_path_length] = True
        # TODO filter out rollouts which are completed

        return observations, dones, rewards #, tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self):
        results = env.reset_batch(self.n)
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass


class VectorizedSampler(BaseSampler):

    """
    A sampler which works with policies and environments which are both
    vectorized / batched.

    Forked from `sandbox.rocky.tf.samplers.vectorized_sampler`.
    """

    def start_worker(self):
        estimated_envs = int(self.algo.batch_size / self.algo.max_path_length)
        estimated_envs = max(1, min(estimated_envs, 100))
        self.vec_env = VecEnvExecutor(
            self.algo.env,
            n=estimated_envs,
            max_path_length=self.algo.max_path_length
        )
        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, itr):
        logger.log("Obtaining samples for iteration %d..." % itr)
        paths = []
        n_samples = 0
        obses = self.vec_env.reset()
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0
        import time
        while n_samples < self.algo.batch_size:
            t = time.time()
            self.algo.policy.reset(dones)
            actions, agent_infos = self.algo.policy.get_actions(obses)
            policy_time += time.time() - t
            t = time.time()
            next_obses, dones, rewards, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in xrange(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in xrange(self.vec_env.num_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths.append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        return paths
