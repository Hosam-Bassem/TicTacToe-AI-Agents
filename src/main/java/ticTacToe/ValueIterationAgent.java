package ticTacToe;


import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=10;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	public void iterate()
	{
		int iteration = 0;
	    double argMaxValue;
	    
	    // perform for k steps
	    while (iteration < k) {
	        Iterator<Game> stateIterator = this.valueFunction.keySet().iterator();
	        
	        // perform value iteration for all states
	        while (stateIterator.hasNext()) {
	        	argMaxValue = Double.NEGATIVE_INFINITY;
	            Game state = stateIterator.next();
	    
	            if (state.isTerminal()) { // Skip terminal States
	                this.valueFunction.put(state, 0.0);
	                continue;
	            }
	    
	            Iterator<Move> moveIterator = state.getPossibleMoves().iterator();
	            // Calculate q values for all possible moves
	            while (moveIterator.hasNext()) {
	                Move move = moveIterator.next();  
	                double qValue = calculateQValue(state, move);  
	                argMaxValue = Math.max(argMaxValue, qValue); // Set V(s) value for the state
	            }
	            // value update: set the new value of the state
	            this.valueFunction.put(state, argMaxValue);
	        }
	        iteration++;
	    }
	}
	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		Policy extractedPolicy = new Policy();
	    Iterator<Game> stateIterator = valueFunction.keySet().iterator();

	    while (stateIterator.hasNext()) {
	    	Game state = stateIterator.next();

	        if (state.isTerminal()) { // skip terminal states
	            valueFunction.put(state, 0.0);
	            continue;
	        }
	        // a map to store Q-values for all possible moves
	        Map<Move, Double> moveQValues = new HashMap<>();

	        Iterator<Move> moveIterator = state.getPossibleMoves().iterator();
	        
	        // Calculate Q-values for each possible move
	        while (moveIterator.hasNext()) {
                Move move = moveIterator.next();  
                double qValue = calculateQValue(state, move);  
                moveQValues.put(move, qValue);
            }
	        
	        // The best move: the move with the highest Q-value
	        Move bestMove = Collections.max(moveQValues.entrySet(), Map.Entry.comparingByValue()).getKey();
	        extractedPolicy.policy.put(state, bestMove); // Update the policy with the best move for the current state
	    }
	    
	    return extractedPolicy;
	}
	
	/**
	 * calculates the Q-value for a given state and move with the transition probabilities.
	 * @param state a given state
	 * @param move a given action
	 * @return the calculated q value using bellman's eqn
	 */
	private double calculateQValue(Game state, Move move) {
	    double qValue = 0; // Value Initialization
	    
	    // getting the transitions for the given state and move
	    List<TransitionProb> transitions = mdp.generateTransitions(state, move); // all possible transitions for the given state and move
	    Iterator<TransitionProb> transitionIterator = transitions.iterator();
	    
	    // Performs q value calculation for a given state with all possible moves
	    while (transitionIterator.hasNext()) {
	        TransitionProb transition = transitionIterator.next();
	        double reward = transition.outcome.localReward;
	        double futureValue = this.valueFunction.get(transition.outcome.sPrime);
	        qValue += transition.prob * (reward + (discount * futureValue)); // Value Updated
	    }
	    
	    return qValue;
	}
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
}
