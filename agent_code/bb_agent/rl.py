import numpy as np
import events as e
import settings as s


STATE_F_LENGTH = 19
GAMMA = 0.7
ALPHA = 0.04
EXPERIENCE_BUFFER_EPISODES = 1000
BATCH_SIZE = 800
SOFTMAX_TEMP_TRAIN = 2
SOFTMAX_TEMP_PLAY = 0.5
EPSILON_TRAIN = 0.33
EPSILON_PLAY = 0.1
CLIP_MIN = 0.00000001

REWARDS = {
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -5,
        e.INVALID_ACTION: -15,
        e.BOMB_DROPPED: -1,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 8,
        e.COIN_FOUND: 12,
        e.COIN_COLLECTED: 30,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -70,
        e.GOT_KILLED: -60,
        e.OPPONENT_ELIMINATED: 20,
        e.SURVIVED_ROUND: 20,
}

MODEL_FILE_NAME = "model.pt"
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def extract_features(state):
    # features: 10^14 states
        # wall above?
        # wall below?
        # wall left?
        # wall right?
        # map border?
        # vec to closest agent
        # oponents present?
        # vec to 2 closest bombs
        # number of bombs present (0, 1, >1)
        # vec to nearest crate
        # crates present?
        # vec to nearest coin
        # coins present?
    
    res = np.zeros(STATE_F_LENGTH)
    own_pos = state['self'][3]

    #wall adjacency
    if state['field'][own_pos[0], own_pos[1]-1] == -1:
        res[0] = 1
    if state['field'][own_pos[0], own_pos[1]+1] == -1:
        res[1] = 1
    if state['field'][own_pos[0]-1, own_pos[1]] == -1:
        res[2] = 1
    if state['field'][own_pos[0]+1, own_pos[1]] == -1:
        res[3] = 1

    #map border
    if (1 in own_pos) or (s.COLS-1 == own_pos[0]) or (s.ROWS-1 == own_pos[1]):
        res[4] = 1

    #nearest agent
    enemy_count = len(state['others'])
    if enemy_count > 0:
        dist_vecs = np.zeros((enemy_count, 2))
        for i in range(enemy_count):
            dist_vecs[i,0] = state['others'][i][3][0] - own_pos[0]
            dist_vecs[i,1] = state['others'][i][3][1] - own_pos[1]
        closest = np.argmin(np.linalg.norm(dist_vecs, axis=1))
        res[5] = dist_vecs[closest, 0]
        res[6] = dist_vecs[closest, 1]
        res[7] = 1

    #bombs
    bomb_count = len(state['bombs'])
    if bomb_count > 0:
        dist_vecs = np.zeros((bomb_count, 2))
        for i in range(bomb_count):
            dist_vecs[i,0] = state['bombs'][i][0][0] - own_pos[0]
            dist_vecs[i,1] = state['bombs'][i][0][1] - own_pos[1]
        indices = np.argsort(np.linalg.norm(dist_vecs, axis=1))
        if bomb_count == 1:
            res[8] = dist_vecs[indices[0], 0]
            res[9] = dist_vecs[indices[0], 1]
            res[12] = 1
        if bomb_count > 1:
            res[10] = dist_vecs[indices[1], 0]
            res[11] = dist_vecs[indices[1], 1]
            res[12] = 2

    #nearest crate
    # TODO: vectorize!
    dist_vecs = []
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if state['field'][x,y] == 1:
                dist_vecs.append([x-own_pos[0], y-own_pos[1]])
    if len(dist_vecs) > 0:
        dist_vecs = np.array(dist_vecs)
        closest = np.argmin(np.linalg.norm(dist_vecs, axis=1))
        res[13] = dist_vecs[closest, 0]
        res[14] = dist_vecs[closest, 1]
        res[15] = 1

    #nearest coin
    coin_count = len(state['coins'])
    if coin_count > 0:
        dist_vecs = np.zeros((coin_count, 2))
        for i in range(coin_count):
            dist_vecs[i,0] = state['coins'][i][0] - own_pos[0]
            dist_vecs[i,1] = state['coins'][i][1] - own_pos[1]
        closest = np.argmin(np.linalg.norm(dist_vecs, axis=1))
        res[16] = dist_vecs[closest, 0]
        res[17] = dist_vecs[closest, 1]
        res[18] = 1
    
    return res


def reward_from_events(events):
    res = 0
    for ev in events:
        if ev in REWARDS:
            res += REWARDS[ev]
    return res


def q_function(state_f, action, beta):
    return np.dot(state_f, beta[ ACTIONS.index(action) ].T)


def y_for_episode(episode_buffer):
    T = len(episode_buffer)
    y = np.zeros(T)
    for t in range(T):
        t_prime = t + 1
        while t_prime < T:
            y[t] += GAMMA**(t_prime - t - 1) * episode_buffer[t_prime][3]
            t_prime += 1
    return y


def update_beta(batch, old_beta):
    new_beta = np.zeros(np.shape(old_beta))

    #sort batch by action
    action_batches = {}
    for a in ACTIONS:
        action_batches[a] = []
    for episode in batch:
        for t in range(len(episode[0])):
            action = episode[0][t][1]
            state_f = episode[0][t][0]
            y = episode[1][t]
            # save x and y for each timestep sorted by action
            action_batches[action].append((state_f, y))

    #update each action
    for a, action in enumerate(ACTIONS):
        sub_batch = action_batches[action]
        if len(sub_batch) > 0:
            sum = np.zeros(STATE_F_LENGTH)
            for elem in sub_batch:
                sum += elem[0].T * (elem[1] - elem[0] * old_beta[a])
            new_beta[a,:] = old_beta[a,:] + ALPHA * sum / len(sub_batch)
    
    return new_beta


def select_action(state, beta, train):
    ret = np.zeros(len(ACTIONS))
    for a, action in enumerate(ACTIONS):
        ret[a] = q_function(extract_features(state), action, beta)
    
    if train:
        return epsilon_greedy_select(ret, train)
    else:
        return softmax_select(ret, train)


def softmax_select(rewards, train):
    if train:
        mult = 1 / SOFTMAX_TEMP_TRAIN
    else:
        mult = 1 / SOFTMAX_TEMP_PLAY
    distrib = np.exp(rewards*mult) / np.sum(np.exp(rewards*mult))
    return np.random.choice(ACTIONS, p=distrib)


def epsilon_greedy_select(rewards, train):
    if train:
        eps = EPSILON_TRAIN
    else:
        eps = EPSILON_PLAY
    if np.random.rand() < eps:
        return ACTIONS[np.random.randint(len(ACTIONS))]
    else:
        return ACTIONS[np.argmax(rewards)]