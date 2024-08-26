from dataclasses import dataclass
import numpy as np
import torch
from prioritized_experience_replay.sum_tree import SumTree


@dataclass
class Batch:

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class PrioritizedReplayBuffer:
    sum_tree: SumTree

    buffer_size: int
    epsilon: float
    alpha: float
    beta: float
    max_priority: float

    real_size: int
    node_idx: int

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        epsilon: float = 0.01,
        alpha: float = 0.1,
        beta: float = 0.1,
    ):
        self.sum_tree = SumTree(size=buffer_size)
        self.buffer_size = buffer_size

        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.max_priority = epsilon

        self.real_size = 0
        self.node_idx = 0

        self.observations = np.zeros(
            (buffer_size, *observation_space.shape), dtype=observation_space.dtype
        )
        action_dim = int(np.prod(action_space.shape))
        self.actions = np.zeros((buffer_size, action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_observations = np.zeros(
            (buffer_size, *observation_space.shape), dtype=observation_space.dtype
        )
        self.dones = np.zeros(buffer_size, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ):
        self.sum_tree.add(self.max_priority, self.node_idx)

        self.observations[self.node_idx] = obs
        self.actions[self.node_idx] = action
        self.rewards[self.node_idx] = reward
        self.next_observations[self.node_idx] = next_obs
        self.dones[self.node_idx] = done

        self.node_idx = (self.node_idx + 1) % self.buffer_size
        self.real_size = min(self.buffer_size, self.real_size + 1)

    # Sampling inspired from https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
    def sample(self, batch_size: int):
        sample_indices = []
        tree_indices = []
        priorities = np.zeros((batch_size, 1), dtype=np.float32)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.sum_tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = np.random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.sum_tree.get(cumsum)

            priorities[i] = priority
            tree_indices.append(tree_idx)
            sample_indices.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.sum_tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = torch.tensor(weights / weights.max())

        batch = Batch(
            torch.from_numpy(self.observations[sample_indices]),
            torch.from_numpy(self.actions[sample_indices]),
            torch.from_numpy(self.rewards[sample_indices]),
            torch.from_numpy(self.next_observations[sample_indices]),
            torch.from_numpy(self.dones[sample_indices]),
        )
        return batch, weights, tree_indices

    def update_priorities(self, data_indices: list[int], priorities: list[float]):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_indices, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.sum_tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
