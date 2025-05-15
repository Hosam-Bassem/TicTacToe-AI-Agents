# TicTacToe: Markov Decision Processes & Reinforcement Learning 
An academic project to implement Value Iteration, Policy Iteration that plan/learn to play 3x3 Tic-Tac-Toe game in Java.

## Files
* **ValueIterationAgent.java** -	A Value Iteration agent for solving the Tic-Tac-Toe game with an assumed MDP model.

* **PolicyIterationAgent.java** -	A Policy Iteration agent for solving the Tic-Tac-Toe game with an assumed MDP model.

* **QLearningAgent.java** -	A q-learner, Reinforcement Learning agent for the Tic-Tac-Toe game.

* **Game.java** -	The 3x3 Tic-Tac-Toe game implementation.

* **TTTMDP.java** -	Defines the Tic-Tac-Toe MDP model

* **TTTEnvironment.java** -	Defines the Tic-Tac-Toe Reinforcement Learning environment

* **Agent.java** - Abstract class defining a general agent, which other agents subclass.

* **HumanAgent.java** - Defines a human agent that uses the command line to ask the user for the next move

* **RandomAgent.java** - Tic-Tac-Toe agent that plays randomly according to a RandomPolicy

* **Move.java** - Defines a Tic-Tac-Toe game move

* **Outcome.java** - A transition outcome tuple (s,a,r,s’)

* **Policy.java** - An abstract class defining a policy – you should subclass this to define your own policies

* **TransitionProb.java** - A tuple containing an Outcome object and a probability of the Outcome occurring.

* **RandomPolicy.java** - A subclass of policy – it’s a random policy used by a RandomAgent instance.

## Run the Program
Run Game.java without any parameters and you’ll be able to play the RandomAgent using the command line. From within the top level, main project folder:
> java –cp target/classes/ ticTacToe.Game

You should be able to win or draw easily against this agent. Not a very good agent!
You can control many aspects of the Game, but mainly which agents will play each other. A full list of options is available by running:
> java –cp target/classes/ ticTacToe.Game -h

Use the –x & -o options to specify the agents that you want to play the game. Value Iteration, Policy Iteration, and Q-Learning agents are denoted as vi, pi & ql respectively, and can only play X in the game. This ignores the problem of dealing with isomorphic state spaces (mapping x’s to o’s and o’s to x’s in this case). For example if you want two RandomAgents to play out the game, you do it like this:
> java target/classes/ ticTacToe.Game –x random –o random

Look at the console output that accompanies playing the game. You will be told about the rewards that the ‘X’ agent receives. The `O’ agent is always assumed to be part of the environment.
