import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
    from Tkinter import PhotoImage
else:
    import tkinter as tk
    from tkinter import PhotoImage

from agent import Agent, AgentMC
import tqdm
import pickle

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Maze')
    
    # agent
    parser.add_argument('--algo', type=str, default='mc', help='Agent to use') # mc, vi
    
    # mode
    parser.add_argument('--mode', type=str, default='train', help='Mode') # train, test
    parser.add_argument('--ckpt', type=str, default='agent.pkl', help='Model checkpoint')

    # hyperparameters
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--episode', type=int, default=200, help='Number of episodes')
    parser.add_argument('--epsilon', type=float, default=0.20, help='Epsilon greedy')
    parser.add_argument('--show', type=bool, default=True, help='Show the maze')
    parser.add_argument('--hot_start', type=int, default=200, help='Number of hot start')
    parser.add_argument('--from_origin', type=bool, default=False, help='Start from origin')
    
    return parser.parse_args()


UNIT = 100   # 迷宫中每个格子的像素大小
MAZE_H = 6  # 迷宫的高度（格子数）
MAZE_W = 6  # 迷宫的宽度（格子数）


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r'] # 决策空间
        self.n_actions = len(self.action_space)
        self.title('Q-learning')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        """
        迷宫初始化
        """
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([UNIT/2, UNIT/2])
        
        self.bm_stone = PhotoImage(file="obstacles.png")
        self.stone1 = self.canvas.create_image(origin[0]+UNIT * 4, origin[1]+UNIT,image=self.bm_stone)
        self.stone2 = self.canvas.create_image(origin[0]+UNIT, origin[1]+UNIT * 4,image=self.bm_stone)
        self.stone3 = self.canvas.create_image(origin[0]+UNIT*4, origin[1]+UNIT * 3,image=self.bm_stone)
        self.stone4 = self.canvas.create_image(origin[0]+UNIT*3, origin[1]+UNIT * 4,image=self.bm_stone)

        self.bm_yoki = PhotoImage(file="character.png")
        self.yoki = self.canvas.create_image(origin[0], origin[1],image=self.bm_yoki)

        self.bm_Candy = PhotoImage(file="candy.png")
        self.Candy = self.canvas.create_image(origin[0]+4*UNIT, origin[1]+4*UNIT,image=self.bm_Candy)

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.yoki)
        origin = np.array([UNIT/2, UNIT/2])
        
        self.yoki = self.canvas.create_image(origin[0], origin[1],image=self.bm_yoki)
        return self.canvas.coords(self.yoki)

    def set_yoki(self, state):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.yoki)
        origin = np.array([UNIT/2, UNIT/2]) + np.array([UNIT * state[0], UNIT * state[1]])
        
        if list(origin) in [self.canvas.coords(self.Candy), 
                            self.canvas.coords(self.stone1), 
                            self.canvas.coords(self.stone2), 
                            self.canvas.coords(self.stone3),
                            self.canvas.coords(self.stone4)]:
            return self.random_yoki()
        
        self.yoki = self.canvas.create_image(origin[0], origin[1],image=self.bm_yoki)
        return self.canvas.coords(self.yoki)

    def random_yoki(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.yoki)
        while True:
            origin = np.array([UNIT/2, UNIT/2]) + np.array([np.random.randint(0, MAZE_W) * UNIT, np.random.randint(0, MAZE_H) * UNIT])
            if list(origin) not in [self.canvas.coords(self.Candy), 
                                    self.canvas.coords(self.stone1), 
                                    self.canvas.coords(self.stone2), 
                                    self.canvas.coords(self.stone3),
                                    self.canvas.coords(self.stone4)]:
                break
        
        self.yoki = self.canvas.create_image(origin[0], origin[1],image=self.bm_yoki)
        return self.canvas.coords(self.yoki)

    def step(self, action):
        s = self.canvas.coords(self.yoki)
        base_action = np.array([0, 0])
        if action == 0:   # 向上移动
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # 向下移动
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # 向右移动
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # 向左移动
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.yoki, base_action[0], base_action[1]) 
        s_ = self.canvas.coords(self.yoki)

        # 回报函数
        if s_ == self.canvas.coords(self.Candy):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.stone1), self.canvas.coords(self.stone2),self.canvas.coords(self.stone3),self.canvas.coords(self.stone4)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False
        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

def update():
    # 更新图形化界面
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

def random_walk():
    s = env.reset()
    while True:
        env.render()
        a = np.random.choice(env.n_actions)
        s, r, done = env.step(a)
        if s != 'terminal':
            state = np.array(s) / UNIT
            state = state.astype(int)
            print('state:', state)
        if done:
            break

def value_iteration():
    agent = Agent(gamma=0.9)
    agent.learn()

    s = env.reset()
    while True:
        
        if s != 'terminal':
            state = np.array(s) / UNIT
            state = state.astype(int)

        env.render()
        a = agent.policy(tuple(state))
        s, r, done = env.step(a)
        if done:
            break

def mc_control():
    agent = AgentMC(gamma = args.gamma, unit = UNIT, size = (MAZE_W, MAZE_H))
    agent.mc_control(env, episode = args.episode, epsilon = args.epsilon, show = args.show, hot_start = args.hot_start, from_origin=args.from_origin)
    agent.save(args.ckpt)

    s = env.reset()

    while True:
        env.render()
        state = agent.get_state(agent.as_tuple(s))
        a = agent.get_action(state)
        s, r, done = env.step(a)
        if done:
            break

def q_learning():
    agent = AgentMC(gamma = args.gamma, unit = UNIT, size = (MAZE_W, MAZE_H))
    print('Q-learning')
    print('gamma:', args.gamma, 'episode:', args.episode, 'epsilon:', args.epsilon, 'show:', args.show, 'hot_start:', args.hot_start, 'from_origin:', args.from_origin)
    agent.q_learning(env, episode = args.episode, epsilon = args.epsilon, show = args.show, hot_start = args.hot_start, from_origin=args.from_origin)
    agent.save(args.ckpt)

    s = env.reset()

    while True:
        env.render()
        state = agent.get_state(agent.as_tuple(s))
        a = agent.get_action(state)
        s, r, done = env.step(a)
        if done:
            break

def on_given():
    agent = AgentMC(gamma = args.gamma, unit = UNIT, size = (MAZE_W, MAZE_H))
    agent.on_given(env)

def main():

    if args.mode == 'train':
        if args.algo == 'mc':
            env.after(100, mc_control)
        elif args.algo == 'q':
            env.after(100, q_learning)
        elif args.algo == 'vi':
            env.after(100, value_iteration)
        elif args.algo == 'og':
            env.after(100, on_given)
        else:
            env.after(100, random_walk)
    else:
        env.after(100, run)


def run():
    agent = pickle.load(open(args.ckpt, 'rb'))

    s = env.reset()

    while True:
        env.render()
        state = agent.get_state(agent.as_tuple(s))
        a = agent.get_action(state)
        s, r, done = env.step(a)
        if done:
            break

if __name__ == '__main__':
    np.random.seed(6)
    args = parse_args()
    env = Maze()
    main()
    env.mainloop()