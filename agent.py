import numpy as np

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
         