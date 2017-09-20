import numpy as np
import pandas as pd
import timeit
from pulp import *

filename = "mdp.csv"
problem_name = filename[:-4]
# data = pd.read_csv("smallsize_test_nr.csv", delimiter=",", header=0)
data = pd.read_csv(filename, delimiter=",", header=0)
n_states = max(max(data['idstatefrom']) + 1, max(data['idstateto']) + 1)
n_actions = max(data['idaction']) + 1
states = [i for i in range(n_states)]
actions = data['idaction'].unique()
stateVals = {key: 0.0 for key in states}
stateActions = {key: 0.0 for key in actions}

P = {tuple(x[:3]): x[3] for x in data[['idstatefrom', 'idaction', 'idstateto', 'probability']].values}
R = {tuple(x[:3]): x[3] for x in data[['idstatefrom', 'idaction', 'idstateto', 'reward']].values}

N = {i: [] for i in range(n_states)}
# for every state k[0] add actions and resulting neighbours
for k in R.keys():
    N[int(k[0])].append([int(k[2]), int(k[1]), R[k[0], k[1], k[2]]])

gamma = 0.9
vars_ = LpVariable.dict("s", stateVals, lowBound=0)
prob = LpProblem("MDP", LpMinimize)

# objective function
prob += lpSum(vars_[i] for i in range(n_states))

# constraints
idx = 0
for s in range(n_states):
    for a in range(n_actions):
        sum_tr = lpSum([gamma * P[s, a, s_] * vars_[s_] for s_ in range(n_states) if (s, a, s_) in P])
        sum_r = lpSum([R[s, a, s_] * P[s, a, s_] for s_ in range(n_states) if (s, a, s_) in R])
        prob += (vars_[s] - sum_tr >= sum_r, "c_" + str(idx))
        idx += 1

start_time = timeit.default_timer()
prob.solve()
elapsed = timeit.default_timer() - start_time

# rew - sold(p-c) - [inventory]_{+}Ch + [inventory]_{-}cb

val_ar = np.zeros(n_states)
policy = np.zeros(n_states)

for v in prob.variables():
    print(v.name, v.varValue)
    val_ar[int(v.name[2:])] = v.varValue

for s in range(n_states):
    max_v = -99999
    best_action = -1
    for n in N[s]:
        v = gamma * P[s, n[1], n[0]] * val_ar[n[0]] + R[s, n[1], n[0]]
        if v > max_v:
            max_v = v
            best_action = n[1]
    policy[s] = best_action

result = pd.DataFrame(
    {
        'idstate': states,
        'idaction': policy
    }
)
result.set_index('idstate')
result.to_csv("./" + problem_name + "_policy.csv", sep=',')
# print(prob.constraints)
print(val_ar)
print(policy)

# MDP TOOLBOX FROM HERE ON
# P = np.zeros((n_actions, n_states, n_states))
# R = np.zeros((n_actions, n_states, n_states))
# print(n_states)
# print(n_actions)
# for index, row in data.iterrows():
#     P[int(row["idaction"]), int(row["idstatefrom"]), int(row["idstateto"])] = float(row["probability"])
#     R[int(row["idaction"]), int(row["idstatefrom"]), int(row["idstateto"])] = float(row["reward"])
#
# for i in range(n_actions):
#     for j in range(n_states):
#         P[i, j, :] /= np.sum(P[i, j, :])
#
# for i in range(n_actions):
#     for j in range(n_states):
#         s = np.sum(P[i, j, :])
#         if s != 1.0:
#             print(i, j, np.sum(P[i, j, :]))
#
# start_time = timeit.default_timer()
# lp = mdptoolbox.mdp._LP(P, R, 0.9)
# lp.run()
# elapsed = timeit.default_timer() - start_time
# print(lp.policy)
# print(len(lp.policy))
# states = [i for i in range(n_states)]
#
# result = pd.DataFrame(
#     {'idaction': lp.policy,
#      'idstate': states,
#      }
# )
# result.sort_values('idstate')
# # print(result)
# print("Time elapsed:")
# print(elapsed)
# print(lp.time)
