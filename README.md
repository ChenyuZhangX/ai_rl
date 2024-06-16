# AI Coding Homework 3: RL

## Problem Formulation

- State Space 

$$\mathcal{S} = \{ (x, y) | x, y \in \{1, 2, 
3, 4, 5, 6\} \}$$

- Action Space 

$$\mathcal{A} = \{ \text{up}, \text{down}, \text{left}, \text{right} \}$$

- Transition Probability 

$$P(s' | s, a) = \begin{cases} 1 & \text{if } s' = \text{next}(s, a) \\ 0 & \text{otherwise} \end{cases}$$

- Reward Function 

$$r(s,a) = \begin{cases} 1 & \text{if } \text{ next}(s,a) == (5, 5) \\ -1 & \text{if } \text{ next}(s,a) \in \text{coord}_{stone} \\ 0 & \text{otherwise} \end{cases}$$

Where $\text{next}(s, a)$ is the next state after taking action $a$ from state $s$, when done is True, the transition ends, $\text{next function}$ gives nothing. And $\text{coord}_{stone}$ is the set of coordinates of the stones.

## Method

- **Value Iteration**: Super effective

- **Monte Carlo Control**: Super slow because it is easy to get stuck by the stones

- **Q learning**: Super slow, also by the stones

## Run

```python
# train the agent
python maze_env.py --mode train --algo vi # Value Iteration, mc, q for Monte Carlo Control, Q learning, og for run on a given track

# test the trained policy, only for mc and q
python maze_env.py --mode test --ckpt agent.pkl 
```




