package ticTacToe;

import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=40000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()
	{
		this(new RandomAgent(), 0.1, 40000, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 */
	
	public void train()
	{
		int episodes = 0;
		
		// loop for the set number of episodes
	    while (episodes < numEpisodes) {
	    	env.reset();
	        Game currentState = env.getCurrentGameState();

	        while (!currentState.isTerminal()) {
	        	List<Move> possibleMoves = currentState.getPossibleMoves();
	            Move selectedMove;

	            if (Math.random() < epsilon) {
	                // Exploration: choose a random move
	            	int moveIndex = (int) (Math.random() * possibleMoves.size());
	                selectedMove = possibleMoves.get(moveIndex);
	            } else {
	                // Exploitation: choose the best move based on the Q-values
	                Map<Move, Double> qValues = getQValuesForAllMoves(currentState);
	                selectedMove = Collections.max(qValues.entrySet(), Map.Entry.comparingByValue()).getKey();
	            }

	            Outcome outcome = null;
	            try {
	                outcome = env.executeMove(selectedMove);
	            } catch (IllegalMoveException e) {
	                e.printStackTrace();
	            }

	            // Update Q-value based on the outcome
	            updateQValue(currentState, selectedMove, outcome);

	            // Move to the next state
	            currentState = outcome.sPrime;
	        }

	        episodes++;
	    }

		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	}
	
	/**
	 * Helper method that retrieves the Q-values for all possible moves in the given state.
	 *
	 * @param state the current state for which Q-values of all possible moves are required.
	 * @return a HashMap containing each possible move as a key and its corresponding Q-value as the value.
	 */
	private HashMap<Move, Double> getQValuesForAllMoves(Game state) {
		HashMap<Move, Double> allMoves = new HashMap<>();  // Initialize a HashMap to store Q-values for all moves

	    if (state.isTerminal()) return allMoves; // if state is empty return an empty hashmap
	    

	    List<Move> moves = state.getPossibleMoves();  // Retrieve all possible moves from the current state
	    
	    // Loop through each move and populates the hash map with its respect Q-value
	    int i = 0;
	    while (i < moves.size()) {
	    	Move move = moves.get(i);
	        allMoves.put(move, qTable.getQValue(state, move));
	        i++;
	    }
	    return allMoves;
	}
	
	/**
	 * Helper method that updates the Q-value for a given state and move based on the outcome of the move.
	 *
	 * @param currentState the current state where the move was taken.
	 * @param selectedMove the move taken in the current state.
	 * @param outcome the outcome object containing the move's result, including the next state and reward.
	 */
	private void updateQValue(Game currentState, Move selectedMove, Outcome outcome) {
		double qVal = qTable.getQValue(outcome.s, outcome.move); // the current Q-value for the state-move pair
	    
	    double argMaxNextQVal = 0.0; // To store the max q value of the next state i.e Q(s', a')
	    
	    // If the next state is not terminal, calculate the maximum Q-value for all possible moves in the next state
	    if (!outcome.sPrime.isTerminal()) {
	    	double maxQForNextState = -Double.MAX_VALUE;
	    	
	    	// Retrieve all possible moves in the next state
	        List<Move> nextPossibleMoves = outcome.sPrime.getPossibleMoves();
	        
	        int i = 0;
	        // Iterate through each move and find the maximum Q-value
	        while (i < nextPossibleMoves.size()) {
	        	Move nextMove = nextPossibleMoves.get(i);
	            double nextQ = qTable.getQValue(outcome.sPrime, nextMove); // Get the Q-value for the next move
	            maxQForNextState = Math.max(maxQForNextState, nextQ); // Update the max Q-Value
	            i++;
	        }
	        argMaxNextQVal = maxQForNextState; // Store the maximum Q-value for the next state
	    }
	    
	    double updatedQ = (1 - alpha) * qVal + alpha * (outcome.localReward + discount * argMaxNextQVal); // Q value calculation
	    qTable.addQValue(outcome.s, outcome.move, updatedQ); // Add the updated Q-value to the Q-table
	}
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		Policy extractedPolicy = new Policy(); // Policy Initialization
		
		// loop through all states
	    for (Game state : qTable.keySet()) {
	        if (state.isTerminal()) continue; // skip terminal states
	        
	        // The optimal is the one with the highest Q-Value for the current state
	        Move optimalMove = Collections.max(getQValuesForAllMoves(state).entrySet(), Map.Entry.comparingByValue()).getKey();
	        extractedPolicy.policy.put(state, optimalMove); // Add the state and its optimal move to the extracted policy
	    }

	    return extractedPolicy; // policy containing the optimal actions for all non-terminal states
	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
	
	
	


	
}
