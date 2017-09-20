# Assignment 1 #

**Due**: 8/30

## Problem 1 ##

Consider the following problem. You are selling *strawberries* and every day you need to purchase a certain quantity from a distributor. Your cost to buy a pint is $`c`$ and the price you sell it for is $`p`$. That means that with every sale you make $`p-c`$. You have to make the purchase decision in the morning before knowing the demand, which is distributed according to a random variable with CDF $`F`$. 

Strawberries spoil quickly and they must be discarded at the end of the day, leading to a loss of value $p$ for each pint. 

Check out the [simple simulator](https://rlrl.shinyapps.io/fruitvendor/) from the class too. (Look for the source code in the folder Code in this repository)
 
The optimal quantity $`q`$ to order in the morning is (see e.g. [Wikipedia](https://en.wikipedia.org/wiki/Newsvendor_model)):
```math
q = F^{-1}\left( \frac{p - c}{p} \right)~.
```
Prove it. 

*Hint*: Using [Leignitz integral rule](https://en.wikipedia.org/wiki/Leibniz_integral_rule), one can show that:
```math
\mathbb{E}[\min\{ X - q, 0 \}] = -F(q)
```

## Problem 2 ##

Now you are selling candy instead. Candy does not spoil and any unsold quantity can be stored and sold the following day. Storing unsold quantity incurrs a holding cost. Does this affect the optimality of the solution above? Formulate the problem as an MDP. Describes states, actions, rewards, probabilities, horizon, discount, and anything else that you think is relevant. Also see Example 1 on Page 9 of ARL.
