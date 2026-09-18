"""Train one shared policy for every household with Stable-Baselines3.

SB3 is single-agent: it expects a vectorised environment of independent copies. The
standard way to use it for a cooperative-competitive market like this one is
*parameter sharing* — one policy network, with every household treated as one slot
of the vector. Each training step then gives the policy one experience per household
from the same tick of the same market, which is both sample-efficient and exactly
the deployment shape: in ``run_pipeline`` one trained policy decides for every
household.

This adapter is that vector, written out directly rather than pulled from
SuperSuit: it is about a page, and owning it keeps the dependency list to what
agents/CLAUDE.md already names.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from agents.rl_env import GridPeerParallelEnv


class SharedPolicyVecEnv(VecEnv):
    """Presents every household of one ``GridPeerParallelEnv`` as a slot of a VecEnv.

    All households live in the same market, so they share one episode: when it
    ends, every slot is done together and the parallel env is reset once. Following
    SB3's contract, the final observation is returned in
    ``info["terminal_observation"]`` and the next episode's first observation takes
    its place.
    """

    def __init__(self, env: GridPeerParallelEnv, seed: int | None = None) -> None:
        """env: the market to train in. seed: seeds the first reset (and so the run)."""
        self.env = env
        self.agent_ids = list(env.possible_agents)
        agent = self.agent_ids[0]
        super().__init__(
            num_envs=len(self.agent_ids),
            observation_space=env.observation_space(agent),
            action_space=env.action_space(agent),
        )
        self._seed = seed
        self._actions: np.ndarray | None = None
        self.episode_savings_eur = 0.0

    def _stack(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([observations[agent] for agent in self.agent_ids])

    def reset(self) -> VecEnvObs:
        """Start an episode; the stored seed applies to the first reset only."""
        observations, _ = self.env.reset(seed=self._seed)
        self._seed = None
        self.episode_savings_eur = 0.0
        return self._stack(observations)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        assert self._actions is not None, "step_async must be called before step_wait"
        action_map = {agent: self._actions[i] for i, agent in enumerate(self.agent_ids)}
        observations, rewards, terminations, truncations, infos = self.env.step(action_map)

        reward_vec = np.array([rewards[a] for a in self.agent_ids], dtype=np.float32)
        self.episode_savings_eur += float(sum(info["savings_eur"] for info in infos.values()))
        done = all(terminations[a] or truncations[a] for a in self.agent_ids)
        info_list: list[dict[str, Any]] = [dict(infos[a]) for a in self.agent_ids]

        if done:
            terminal = self._stack(observations)
            for i, info in enumerate(info_list):
                info["terminal_observation"] = terminal[i]
                info["TimeLimit.truncated"] = False
                info["episode_savings_eur"] = self.episode_savings_eur
            next_obs = self.reset()
        else:
            next_obs = self._stack(observations)

        dones = np.full(self.num_envs, done, dtype=bool)
        return next_obs, reward_vec, dones, info_list

    def close(self) -> None:
        self.env.close()

    # SB3 probes these on wrappers; one shared market has no per-slot sub-envs.
    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        return [getattr(self.env, attr_name)] * len(self._indices(indices))

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(
        self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any
    ) -> list[Any]:
        result = getattr(self.env, method_name)(*method_args, **method_kwargs)
        return [result] * len(self._indices(indices))

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        return [False] * len(self._indices(indices))

    def _indices(self, indices: Any) -> Sequence[int]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices
