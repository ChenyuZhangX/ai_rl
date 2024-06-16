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

    def afore_has(self, state, action = None, t = None):
        if not action:
            if state in self.state:
                return True
            return False

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

class State:
    def __init__(self, pos: tuple, size: tuple, init_value = 0):
        self.size = size
        self.pos = pos
        self.init_value = init_value
        self.get_actions()
        self.dof = len(self.actions)


    def get_actions(self):
        self.actions = []
        for action in [0, 1, 2, 3]:
            next_state = self.next(action)
            if next_state[0] < 0 or next_state[0] >= self.size[0]:
                continue
            if next_state[1] < 0 or next_state[1] >= self.size[1]:
                continue

            self.actions.append(action)
        self.pi = np.ones(len(self.actions)) / len(self.actions)
        self.q = np.ones(len(self.actions)) * self.init_value
        
    def next(self, action: int):
        if action == 0: # up
            next_state = (self.pos[0], self.pos[1] - 1)
        elif action == 1: # down
            next_state = (self.pos[0], self.pos[1] + 1)
        elif action == 2: # right
            next_state = (self.pos[0] + 1, self.pos[1])
        elif action == 3:
            next_state = (self.pos[0] - 1, self.pos[1])
        else:
            raise ValueError('Invalid action')
        
        return next_state
    
    def __eq__(self, other):
        return self.pos == other.pos
    
    def __hash__(self):
        return hash(self.pos)
        
        

class AgentMC:
    def __init__(self, gamma = 0.9, size = (6, 6), unit = 100):

        self.unit = unit
        self.gamma = gamma
        self.size = size

        self.state_space = [State((i, j), size) for i in range(size[0]) for j in range(size[1])]
        self.Returns = {(state, action): [] for state in self.state_space for action in state.actions}

    def get_action(self, state: State, greedy = False):
        # print(state.pi, state.actions, state.q)
        return np.random.choice(state.actions, p = state.pi) if not greedy else np.argmax(state.q)
    
    def get_state(self, pos: tuple):
        try:
            state = [state for state in self.state_space if state.pos == pos][0]
            return state
        except:
            raise ValueError('No such state')
        
    
    def as_tuple(self, s):
        assert type(s) != 'str', 'Invalid state'
        state = np.array(s) / self.unit
        state = state.astype(int)
        return tuple(state)
    
    def mc_control(self, env, episode = 20, epsilon = 0.20, show = False, hot_start = 0, from_origin = False):
        
        track = [(5, 5), (5, 4), (4, 5), (3, 5), (5, 3), (5, 2), (2, 5)]
        
        total = episode + hot_start

        for epso in tqdm.tqdm(range(total)):
            if hot_start > 0:
                start = np.random.choice(len(track))
                s = env.set_yoki(track[start])
                print(f"Hot start at {track[start]}")
            elif from_origin:
                s = env.reset()
                print(f"Initiate at {self.as_tuple(s)}")
            else:                
                s = env.random_yoki()
                print(f"Initiate at {self.as_tuple(s)}")
            
            chain = MC()

            
            skip = False
            while True:
                if show:
                    env.render()
                
                

                state = self.get_state(self.as_tuple(s))
                a = self.get_action(state)

                ''' Skip the state if it has been visited before
                if chain.afore_has(state):
                    skip = True
                    for a in state.actions:
                        if not chain.afore_has(state, a, len(chain)):
                            skip = False
                            a = a
                            break
                            
                if skip:
                    break
                '''

                s, r, done = env.step(a)
                chain.append(state, a, r)

                if done:
                    break

            if skip:
                print("Skip")
                continue

            hot_start -= 1

            G = 0
            for t in range(len(chain) - 1, -1, -1):
                state, action, reward, _ = chain[t]
                G = self.gamma * G + reward
                if not chain.afore_has(state, action, t):
                    self.Returns[(state, action)].append(G)

                    for idx, a in enumerate(state.actions):
                        state.q[idx] = np.mean(self.Returns[(state, a)])
                    
                    if epso > 100:
                        qs = np.array([state.q[idx] for idx in range(state.dof)])
                        state.pi = np.ones(state.dof) * epsilon / state.dof
                        state.pi[np.argmax(qs)] = 1 - epsilon + epsilon / state.dof

            print(f"Episode {epso + 1} Reward: {G} length: {len(chain)}")

    def q_learning(self, env, episode = 20, epsilon = 0.20, show = False, lr = 0.1, hot_start = 0, from_origin = False):
        track = [(5, 5), (5, 4), (4, 5), (3, 5), (5, 3), (5, 2), (2, 5)]
        total = episode + hot_start
        for episo in tqdm.tqdm(range(total)):
            
            if hot_start > 0:
                start = np.random.choice(len(track))
                s = env.set_yoki(track[start])
                print(f" Hot start at {track[start]}")
            elif from_origin:
                s = env.reset()
                print(f" Initiate at {self.as_tuple(s)}")
            else:                
                s = env.random_yoki()
                print(f" Initiate at {self.as_tuple(s)}")
            
            rewards = []
            while True:
                if show:
                    env.render()
                
                state = self.get_state(self.as_tuple(s))
                a = self.get_action(state)
                s, r, done = env.step(a)

                rewards.append(r)

                if done:    
                    break
                
                # update q
                next_state = self.get_state(self.as_tuple(s))
                next_q = np.max(next_state.q)

                state.q[state.actions.index(a)] += lr * (r + self.gamma * next_q - state.q[state.actions.index(a)])

                s = (next_state.pos[0] * self.unit, next_state.pos[1] * self.unit)

            reward = 0
            for r in rewards[::-1]:
               reward = self.gamma * reward + r
            
            hot_start -= 1

            # update policy
            for state in self.state_space:
                qs = np.array([state.q[idx] for idx in range(state.dof)])
                state.pi = np.ones(state.dof) * epsilon / state.dof
                state.pi[np.argmax(qs)] = 1 - epsilon + epsilon / state.dof

            print(f"Episode {episo + 1} Reward: {reward} length: {len(rewards)}")


    def on_given(self, env):
        track = [1, 1, 1, 1, 1, 2]
        s = env.reset()
        for a in track:
            env.render()
            s, r, done = env.step(a)


    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

if __name__ == '__main__':
    agent = AgentMC()
    for state in agent.state_space:
        print(state.pos, state.actions)
         