from tensorforce import Agent, Environment
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
import pandas as pd
from itertools import combinations


def generate(size, end, difficulty=.3):
    mz = np.zeros((size[0] - 2, size[1] - 2))
    while sum(sum(mz)) / ((mz.shape[0]) * (mz.shape[1])) < difficulty:
        mz[np.random.randint(0, mz.shape[0]), np.random.randint(0, mz.shape[1])] = 1
    mz = binary_erosion(binary_dilation(mz), iterations=2)
    maze = np.ones(size)
    maze[1:-1, 1:-1] = mz
    maze[end] = -1
    return maze


class Maze(Environment):

    def __init__(self):
        super().__init__()
        self.pos = (1, 1)
        self.end = (25, 25)
        self.maze = generate((30, 30), self.end, .3)
        self.steps = 0

    def states(self):
        return dict(type='float', shape=(8,))

    def actions(self):
        return dict(type='int', num_values=4)

    def execute(self, actions):
        act = actions
        valid = True
        done = False
        dx = np.linalg.norm(np.array(self.end) - np.array(self.pos))
        self.steps += 1
        if act == 0:
            if self.maze[self.pos[0] + 1, self.pos[1]] < 1:
                self.pos = (self.pos[0] + 1, self.pos[1])
            else:
                valid = False
        if act == 1:
            if self.maze[self.pos[0] - 1, self.pos[1]] < 1:
                self.pos = (self.pos[0] - 1, self.pos[1])
            else:
                valid = False
        if act == 2:
            if self.maze[self.pos[0], self.pos[1] - 1] < 1:
                self.pos = (self.pos[0], self.pos[1] - 1)
                dx += .1
            else:
                valid = False
        if act == 3:
            if self.maze[self.pos[0], self.pos[1] + 1] < 1:
                self.pos = (self.pos[0], self.pos[1] + 1)
                dx += .1
            else:
                valid = False
        if self.maze[self.pos] == -1:
            done = True
            reward = 50 * (1 - self.steps / 500)
        else:
            if valid:
                reward = dx - np.linalg.norm(np.array(self.end) - np.array(self.pos))
            else:
                reward = -5
        state = np.array([self.maze[self.pos[0] + 1, self.pos[1]],
                          self.maze[self.pos[0] - 1, self.pos[1]],
                          self.maze[self.pos[0], self.pos[1] - 1],
                          self.maze[self.pos[0], self.pos[1] + 1],
                          self.pos[0], self.pos[1],
                          np.sign(self.end[0] - self.pos[0]), np.sign(self.end[1] - self.pos[1])])
        mask = [state[n] != 1 for n in range(4)]
        return dict(state=state, action_mask=mask), done, reward

    def reset(self):
        self.pos = (1, 1)
        self.steps = 0
        state = np.array([self.maze[self.pos[0] + 1, self.pos[1]],
                          self.maze[self.pos[0] - 1, self.pos[1]],
                          self.maze[self.pos[0], self.pos[1] - 1],
                          self.maze[self.pos[0], self.pos[1] + 1],
                          self.pos[0], self.pos[1],
                          np.sign(self.end[0] - self.pos[0]), np.sign(self.end[1] - self.pos[1])])
        mask = [state[n] != 1 for n in range(4)]
        return dict(state=state, action_mask=mask)


def shuffle(l):
    p = np.random.permutation(np.arange(len(l)))
    return [l[n] for n in p]


class Splendor(Environment):
    decks = None
    board = None
    reserves = None
    p_tokens = None
    b_tokens = None
    p_gems = None
    BLANK = [-1, 0, 0, 0, 0, 0, 0]
    curr_player = 0
    state = None
    n_players = 2

    def __init__(self):
        super().__init__()
        act = {}
        n_act = 1

        # Get all possible actions for players
        act[0] = (-1, 0)
        for tile_comb in combinations([0, 1, 2, 3, 4], 3):
            act[n_act] = (2, tile_comb)
            n_act += 1
        for tt in [0, 1, 2, 3, 4]:
            act[n_act] = (2, (tt, tt))
            n_act += 1
        for tnum in range(3):
            for cnum in range(4):
                act[n_act] = (0, cnum + tnum * 4)
                n_act += 1
                act[n_act] = (1, cnum + tnum * 4)
                n_act += 1
        for n in range(3):
            act[n_act] = (3, n)
            n_act += 1
        self.ACTIONS = act
        self.n_act = n_act
        self.reset()

    def states(self):
        # Breakdown as follows:
        # 0-11 is the board cards
        # 12-14 is the reserve
        # 15 is the player's tokens
        # 16 is the player's gems
        # 17 is the board's tokens remaining
        return dict(type='float', shape=(13 + 5 * self.n_players, 7))

    def actions(self):
        return dict(type='int', num_values=self.n_act)

    def execute(self, actions):
        player = self.curr_player
        done = False
        reward = 0.1
        valid = self.validate(actions, player)
        if valid:
            act = self.ACTIONS[actions]
            if act[0] == -1:
                reward = -5
            if act[0] == 0:
                # This is a buy action
                sel_card = self.board[act[1]]
                gold_used = 0
                for n in range(5):
                    tok_used = max(sel_card[n] - self.p_gems[player][n], 0)
                    if tok_used > self.p_tokens[player][n]:
                        tok_used = self.p_tokens[player][n]
                        gold_used += sel_card[n] - self.p_gems[player][n] - self.p_tokens[player][n]
                    self.p_tokens[player][n] -= tok_used
                    self.b_tokens[n] += tok_used
                self.p_tokens[player][5] -= gold_used
                self.b_tokens[5] += gold_used
                self.p_gems[player][int(sel_card[5])] += 1
                self.p_gems[player][5] += sel_card[6]
                reward += sel_card[6] + .1
                try:
                    self.board[act[1]] = self.decks[act[1] // 4 + 1].pop()
                except IndexError:
                    self.board[act[1]] = self.BLANK
            if act[0] == 1:
                # This is a reserve action
                sel_card = self.board[act[1]]
                n_res = [n for n in range(3) if self.reserves[player * 3 + n][0] == -1][0]
                self.reserves[player * 3 + n_res] = sel_card
                self.p_tokens[player][5] += 1
                self.b_tokens[5] -= 1
                reward += .2
                try:
                    self.board[act[1]] = self.decks[act[1] // 4 + 1].pop()
                except IndexError:
                    self.board[act[1]] = self.BLANK
            if act[0] == 2:
                # This is a token taking action
                for tok in act[1]:
                    self.p_tokens[player][tok] += 1
                    self.b_tokens[tok] -= 1
                reward += .1
            if act[0] == 3:
                # Buy a gem from the reserve
                sel_card = self.reserves[player * 3 + act[1]]
                gold_used = 0
                for n in range(5):
                    tok_used = max(sel_card[n] - self.p_gems[player][n], 0)
                    if tok_used > self.p_tokens[player][n]:
                        tok_used = self.p_tokens[player][n]
                        gold_used += sel_card[n] - self.p_gems[player][n] - self.p_tokens[player][n]
                    self.p_tokens[player][n] -= tok_used
                    self.b_tokens[n] += tok_used
                self.p_tokens[player][5] -= gold_used
                self.b_tokens[5] += gold_used
                self.p_gems[player][int(sel_card[5])] += 1
                self.p_gems[player][5] += sel_card[6]
                self.reserves[player * 3 + act[1]] = self.BLANK
                reward += sel_card[6] + .1
        else:
            reward = -5

        # We have to make sure <12 tokens
        '''
        if sum(self.p_tokens[player]) > 12:
            reward -= .1
            while sum(self.p_tokens[player]) > 12:
                for n in range(5):
                    if self.p_tokens[player][n] > 0 and sum(self.p_tokens[player]) > 12:
                        rtoks = min(self.p_tokens[player][n], sum(self.p_tokens[player]) - 12)
                        self.p_tokens[player][n] -= rtoks
                        self.b_tokens[n] += rtoks
        '''

        if self.p_gems[player][5] >= 15:
            done = True
            reward += 10

        self.state = np.array(self.board + self.reserves + self.p_tokens + self.p_gems + [self.b_tokens])

        # Now we generate the mask
        nplayer = player + 1 if player + 1 < self.n_players else 0
        mask = [self.validate(n, nplayer) for n in range(self.n_act)]
        self.curr_player = nplayer
        return dict(state=self.state, action_mask=mask), done, reward

    def validate(self, actions, player):
        valid = True
        act = self.ACTIONS[actions]
        if act[0] == 0:
            # This is a buy action
            sel_card = self.board[act[1]]
            if sel_card[0] == -1:
                valid = False
            else:
                diffs = [self.p_tokens[player][n] + self.p_gems[player][n] - sel_card[n]
                         if self.p_tokens[player][n] + self.p_gems[player][n] - sel_card[n] < 0 else 0
                         for n in range(5)]
                if sum(diffs) + self.p_tokens[player][5] < 0:
                    valid = False
        if act[0] == 1:
            # This is a reserve action
            n_res = [n for n in range(3) if self.reserves[player * 3 + n][0] == -1][0]
            if n_res == 2:
                valid = False
            if self.b_tokens[5] == 0:
                valid = False
        if act[0] == 2:
            # This is a token taking action
            res_toks = self.b_tokens.copy()
            for tok in act[1]:
                res_toks[tok] -= 1
            if np.any([r < 0 for r in res_toks]):
                valid = False
        if act[0] == 3:
            # Buy a gem from the reserve
            if self.reserves[player * 3 + act[1]][0] == -1:
                valid = False
            else:
                sel_card = self.reserves[player * 3 + act[1]]
                diffs = [self.p_tokens[player][n] + self.p_gems[player][n] - sel_card[n]
                         if self.p_tokens[player][n] + self.p_gems[player][n] - sel_card[n] < 0 else 0
                         for n in range(5)]
                if sum(diffs) + self.p_tokens[player][5] < 0:
                    valid = False
        return valid

    def reset(self, num_parallel=None):
        # First, build the deck and shuffle it
        cards = pd.read_csv('/home/jeff/repo/test.txt').fillna(0)
        cards['color'] = cards['color'].map({'black': 2, 'white': 0, 'red': 3,
                                             'blue': 1, 'green': 4})
        self.decks = {1: [], 2: [], 3: []}
        for idx, row in cards.iterrows():
            self.decks[row['tier']].append([row['cost_white'], row['cost_blue'], row['cost_black'], row['cost_red'],
                                            row['cost_green'], row['color'], row['points']])
        for tier in self.decks:
            self.decks[tier] = shuffle(self.decks[tier])

        # Deal out the board
        # Should always be 12 cards
        # t1: 0-3 t2: 4-7 t3: 8-11
        self.board = []
        for tier in self.decks:
            for n in range(4):
                self.board.append(self.decks[tier].pop())

        # Reset player things
        self.p_tokens = [[0, 0, 0, 0, 0, 0, 0] for _ in range(self.n_players)]
        self.p_gems = [[0, 0, 0, 0, 0, 0, 0] for _ in range(self.n_players)]
        self.b_tokens = [4 + 2 * self.n_players, 4 + 2 * self.n_players, 4 + 2 * self.n_players,
                         4 + 2 * self.n_players, 4 + 2 * self.n_players, 4 + 2 * self.n_players, 0]
        self.reserves = []
        for _ in range(self.n_players):
            self.reserves = self.reserves + [self.BLANK for _ in range(3)]

        self.state = np.array(self.board + self.reserves + self.p_tokens + self.p_gems + [self.b_tokens])
        mask = [self.validate(n, 0) for n in range(self.n_act)]
        self.curr_player = 0
        return dict(state=self.state, action_mask=mask)

    def readAction(self, num):
        act = self.ACTIONS[num]
        if act[0] == 0:
            print('Buy')
        elif act[0] == 1:
            print('Reserve')
        elif act[0] == 2:
            print('Tokens')
        elif act[0] == 3:
            print('Reserve Buy')
        else:
            print('Cop out')

    def gameState(self, inp_st=None):
        enum_d = {0: 'W', 1: 'U', 2: 'B', 3: 'R', 4: 'G'}
        st = self.state.astype(int) if inp_st is None else inp_st.astype(int)
        print('GAME BOARD')
        print('W\tU\tB\tR\tG\tClr\tPts')
        for n in range(12):
            print(f'{st[n, 0]}\t{st[n, 1]}\t{st[n, 2]}\t{st[n, 3]}\t{st[n, 4]}\t' + enum_d[st[n, 5]] + f'\t{st[n, 6]}\t')
        print('Tokens')
        yy = 12 + 5 * self.n_players
        print(f'W: {st[yy, 0]}\tU: {st[yy, 1]}\tB: {st[yy, 2]}\tR: {st[yy, 3]}\tG: {st[yy, 4]}\tGold: {st[yy, 5]}')
        print('PLAYER DATA')
        for p in range(self.n_players):
            print('Player {}'.format(p))
            print('RESERVE')
            print('W\tU\tB\tR\tG\tClr\tPts')
            for n in range(12 + 3 * p, 15 + 3 * p):
                if st[n, 0] == -1:
                    print('N/A')
                else:
                    print(f'{st[n, 0]}\t{st[n, 1]}\t{st[n, 2]}\t{st[n, 3]}\t{st[n, 4]}\t' +
                          enum_d[st[n, 5]] + f'\t{st[n, 6]}\t')
            print('TOKENS')
            yy = 12 + 3 * self.n_players + p
            print(f'W: {st[yy, 0]}\tU: {st[yy, 1]}\tB: {st[yy, 2]}\tR: {st[yy, 3]}\tG: {st[yy, 4]}\tGold: {st[yy, 5]}')
            print('GEMS')
            yy = 12 + 4 * self.n_players + p
            print(f'W: {st[yy, 0]}\tU: {st[yy, 1]}\tB: {st[yy, 2]}\tR: {st[yy, 3]}\tG: {st[yy, 4]}')
            print(f'Points: {st[yy, 5]}')


class Yahtzee(Environment):
    def __init__(self):
        super().__init__()
        self.scorecard = np.zeros((13,))
        self.is_scored = np.array([False for _ in range(13)])
        self.dice = [np.random.randint(1, 6) for _ in range(5)]
        self.rnd = 0

    def states(self):
        return dict(type='float', shape=(31,))

    def actions(self):
        return dict(roll=dict(type='bool', shape=(5,)),
                    select=dict(type='int', num_values=13))

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        self.scorecard = np.zeros((13,))
        self.is_scored = np.array([False for _ in range(13)])
        self.dice = [np.random.randint(1, 6) for _ in range(5)]
        self.rnd = 0
        state = dict(state=np.concatenate((self.dice, self.scorecard, self.is_scored)),
                     action_mask=[True for _ in range(5)] + [True for _ in range(13)])
        return state

    def execute(self, actions):
        if self.rnd < 3:
            for d in actions:
                self.dice[d] = np.random.randint(1, 6)
        else:
            self.is_scored[actions] = True
            self.scorecard[actions] = sum(self.dice)
        self.rnd = self.rnd + 1 if self.rnd == 3 else 0

        terminal = False if np.any(self.is_scored) else True
        reward = np.random.random()
        state = dict(state=np.concatenate((self.dice, self.scorecard, self.is_scored)),
                     action_mask=[True for _ in range(5)] + list(np.logical_not(self.is_scored)))
        return state, terminal, reward