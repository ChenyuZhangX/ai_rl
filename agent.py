import numpy as np
import tqdm

class Agent:
    def __init__(self, gamma = 0.9, size = (6, 6)):
        # list of tuples
        self.state_space = [(i, j) for i in range(size[0]) for j in range(size[1])]
        # list
        self.action_space = [0, 1, 2, 3]

        # value function 2D array
        self.v = np.zeros(size)
        # action-value function 3D array
        self.q = np.zeros(size + (4,))
        # discount factor
        self.gamma = gamma
        # policy 2D array
        self.pi = np.zeros(size).astype(int)

        self.candy = (4, 4)
        self.stones = [(4, 1), (4, 3), (1, 4), (3, 4)]
    
    def learn(self, theta = 0.01):
        delta = 0
        while True:
            delta = 0
            new_v = np.zeros_like(self.v)
            for state in self.state_space:
                i, j = state
                if state in self.stones:
                    new_v[i, j] = 0
                    continue
                if state == self.candy:
                    new_v[i, j] = 0
                    continue
                
                v = self.v[i, j]
                new_v[i, j] = np.max(
                    np.array([self.reward((i, j), a) + self.gamma * self.v[self.next((i, j), a)]for a in range(4)])
                    )
                delta = max(delta, abs(v - new_v[i, j]))
            
            self.v = new_v

            if delta < theta:
                break

        for state in self.state_space:
            i, j = state
            if state in self.stones:
                self.pi[i, j] = -1
                continue
            if state == self.candy:
                self.pi[i, j] = -1
                continue
            self.pi[i, j] = np.argmax(
                np.array([self.reward((i, j), a) + self.gamma * self.v[self.next((i, j), a)] for a in range(4)])
                )

    def policy(self, state):
        return self.pi[state]

    def next(self, state: tuple, action: int):
        if action == 0: # up
            next_state = (state[0], state[1] - 1)
        elif action == 1: # down
            next_state = (state[0], state[1] + 1)
        elif action == 2: # right
            next_state = (state[0] + 1, state[1])
        elif action == 3:
            next_state = (state[0] - 1, state[1])
        else:
            raise ValueError('Invalid action')
        
        if next_state[1] < 0 or next_state[1] >= self.v.shape[1]:
            return state
        elif next_state[0] < 0 or next_state[0] >= self.v.shape[0]:
            return state
        else:
            return next_state
        
    def reward(self, state: tuple, action: int):
        next_state = self.next(state, action)
        next_state = tuple(next_state)
        if next_state == self.candy:
            return 1
        elif next_state in self.stones:
            return -1
        else:
            return 0

# Markov Chain  
class MC:
    def __init__(self):
        self.state = []
        self.action = []
        self.reward = []

    def append(self, state, action, reward):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)

    def afore_has(self, state, action, t):
        for i in range(t):
            if self.state[i] == state and self.action[i] == action:
                return True
        return False

    def __len__(self):
        return len(self.state)
    
    def __getitem__(self, index):
        return self.state[index], self.action[index], self.reward[index], index
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

# Monte Carlo Agent        
class MCAgent:
    def __init__(self, gamma = 0.9, size = (6, 6), unit = 100):
        # list of tuples
        self.state_space = [(i, j) for i in range(size[0]) for j in range(size[1])]
        # list
        self.action_space = [0, 1, 2, 3]

        # action-value function 3D array
        self.q = np.zeros(size + (4,))
        # discount factor
        self.gamma = gamma

        # policy 2D array
        self.pi = np.ones(size + (4,)) / 4
        
        # unit of the maze
        self.unit = unit
        # size of the maze
        self.size = size

        self.Returns = {(s, a): [] for s in self.state_space for a in self.action_space}

    def policy(self, state: tuple):
        # sample an action from the policy
        return np.random.choice(self.action_space, p = self.pi[state])
    
    def as_tuple(self, s): # turn maze state to tuple
        assert type(s) != str, 'state should not be terminal state'
        state = np.array(s) / self.unit
        state = state.astype(int)
        return tuple(state)
    
    def mc_control(self, env, episode = 20, epsilon = 0.20, show = False):
        # for each episode
        for _ in tqdm.tqdm(range(episode)):
        
            s = env.random_yoki()
            chain = MC()
            steps = 0
            while True:
                if show:
                    env.render()
                
                state = self.as_tuple(s)
                a = self.policy(state)
                s, r, done = env.step(a)
                chain.append(state, a, r)
                if done:
                    break

            G = 0
            # Reverse the chain
            for t in range(len(chain) - 1, -1, -1):
                state, action, reward, _ = chain[t]
                G = self.gamma * G + reward
                if not chain.afore_has(state, action, t):
                    self.Returns[(state, action)].append(G)
                    self.q[state + (action, )] = np.mean(self.Returns[(state, action)])
                    
                    # update policy
                    qs = np.array([self.q[state + (a, )] for a in self.action_space])
                    self.pi[state] = np.ones(4) * epsilon / 4
                    self.pi[state + (np.argmax(qs), )] = 1 - epsilon + epsilon / 4
    
    


if __name__ == '__main__':
    agent = Agent()
    state_space = [(i, j) for i in range(6) for j in range(6)]
    action_space = [0, 1, 2, 3]
    '''
    for state in state_space:
        for action in action_space:
            print(agent.reward(state, action))
            print(agent.next(state, action))
            print(agent.q[state + (action, )])
            qs = np.array([agent.reward(state, a) for a in action_space]) + agent.q[state[0], state[1], :]
            print(qs)
            print(np.max(qs))
    '''
    agent.learn()
         