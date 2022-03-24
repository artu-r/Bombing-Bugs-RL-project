import numpy as np

"""
This file is just here to store different feature extraction functions.
Just copy one of them into extract_features() in rl.py
"""

def e19(state):
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


def coin_heaven(state):
    # features: (coin heaven only)  10^4 states
        # wall above?
        # wall below?
        # wall left?
        # wall right?
        # map border?
        # dist vector to nearest coin

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

    #nearest coin
    coin_count = len(state['coins'])
    if coin_count > 0:
        dist_vecs = np.zeros((coin_count, 2))
        for i in range(coin_count):
            dist_vecs[i,0] = state['coins'][i][0] - own_pos[0]
            dist_vecs[i,1] = state['coins'][i][1] - own_pos[1]
        closest = np.argmin(np.linalg.norm(dist_vecs, axis=1))
        res[5] = dist_vecs[closest, 0]
        res[6] = dist_vecs[closest, 1]
    
    return res