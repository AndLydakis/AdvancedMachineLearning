# Imagine that you are looking for the perfect rental apartment.
# You have clearly formulated your preferences in terms of the size, location, condition,
# and price of the apartment. Using your preferences, you are able to assign a score (such as a real
# number in the open interval (0-1)) to each apartment that you see and you would like to find one with
# the maximal score. The challenge is that the market is very competitive and if you want to get the
# particular apartment, you have to make an offer right after you see it. There is also no way of predicting
# the score that you'd assign to apartments that will come on the market later.
# You want to make a decision after seeing 5 apartments.
#
# Formulate this problem as an MDP and compute the optimal policy and value function.
import random
import numpy as np


def generate_house(n, ws, wl, wc, wp):
    house_values = []
    for i in range(n):
        s = random.uniform(0, 1)
        l = random.uniform(0, 1)
        c = random.uniform(0, 1)
        p = random.uniform(0, 1)
        house_values.append(s * ws + l * wl + c * wc + p * wp)
    return house_values


def wiki_probs(n):
    probs = np.zeros(n)
    for r in range(1, n + 1):
        if r == 1:
            probs[0] = 1.0 / n
        else:
            sum_ = 0.0
            for i in range(r, n + 1):
                sum_ += (1.0 / (i - 1.0))
            probs[r - 1] = float(r - 1.0) / float(n) * float(sum_)
    return probs


def mdp_probs(s):
    return [1.0 / (s + 1), s / (s + 1)]


class RentalMDP:
    def pick_action(self, t, scores):
        return 0

    def take_action(self):
        return 10

    def __init__(self, ws=0.15, wl=0.25, wc=0.1, wp=0.5, n=3, iter=1000, gamma=1.0, theta=0.001):
        states = np.zeros([2, n])
        states[1, n - 1] = 1.0
        wprobs = wiki_probs(n)
        while True:
            delta = 0.0

            houses = generate_house(n, ws, wl, wc, wp)

            best_prob_idx = np.argmax(wprobs)
            best_prob = max(wprobs)
            best_actual_idx = houses.index(max(houses))
            current_max = houses[0]
            states_ = states[:]

            for t in range(n):
                # probability that the next is the best observed so far
                p1 = 1.0 / (t + 1.0)
                # probability that the next is not the best observed so far
                p0 = t / (t + 1.0)
                # probability that the next is the best overall
                p_ = t / float(n)
                # probability to transfer to end state

                # reward for stopping at the best object observed so far
                rb = t / n
                # reward for stopping when a better has been observed
                rnb = 0.0
                # reward for continuing
                rc = 0.0

                # UPDATE THE (0, t) cases
                # only viable option is to continue
                if t == n - 1:
                    states_[0, t] = 0
                    # UPDATE THE (1, t) cases
                    # we can continue or stop
                    states_[1, t] = 1/n
                    delta1 = abs(states[1, t] - states_[1, t])
                    delta2 = abs(states[0, t] - states_[0, t])
                    delta = max(delta, delta1, delta2)
                else:
                    states_[0, t] = rc + p1 * states[1, t + 1] + p0 * states[0, t + 1]
                    # UPDATE THE (1, t) cases
                    # we can continue or stop
                    states_[1, t] = max(p_, rc + p1 * states[1, t + 1] + p0 * states[0, t + 1])
                    delta1 = abs(states[1, t] - states_[1, t])
                    delta2 = abs(states[0, t] - states_[0, t])
                    delta = max(delta, delta1, delta2)

            if delta < theta:
                print(states_)
                print(wprobs)
                exit()


if __name__ == '__main__':
    RentalMDP()
