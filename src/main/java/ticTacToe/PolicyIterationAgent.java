package ticTacToe;


import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{
	    Iterator<Game> stateIterator = this.policyValues.keySet().iterator();
	    
	    // loop through all the states
	    while (stateIterator.hasNext()) {
	        Game state = stateIterator.next();

	        if (state.isTerminal()) continue; // skip terminal states
	        curPolicy.put(state, state.getPossibleMoves().get((int) (Math.random() * state.getPossibleMoves().size()))); // Policy initialisation
	    }
	}
	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the currrent policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)
	{
		double qValue;
	    double previousValue;
	    
	    Iterator<Game> stateIterator = this.policyValues.keySet().iterator();
	    
	    // loop through all the states
	    while (stateIterator.hasNext()) {
	        Game state = stateIterator.next();
	        
	        if (state.isTerminal()) { // if state is terminal skip it
	            this.policyValues.put(state, 0.0);
	            continue;
	        }

	        previousValue = this.policyValues.get(state); // Store the current value of the state to check for convergence later

	        boolean converged = false;
	        while (!converged) {
	            
	            qValue = calculateQValue(state, this.curPolicy.get(state));  // Q value calculation

	            this.policyValues.put(state, qValue); // Update the value of the state with the newly calculated Q-value

	            // Check for convergence 
	            if (qValue - previousValue <= delta) converged = true;
	            
	            previousValue = qValue; // Update the previous value to the newly computed value for the next iteration
	        }
	    }
	}
		
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
		// Store the current policy to compare with the updated policy later
		Policy previousPolicy = new Policy();
	    Iterator<Map.Entry<Game, Move>> entryIterator = this.curPolicy.entrySet().iterator();
	    double qValue;

	    while (entryIterator.hasNext()) {
	        Map.Entry<Game, Move> entry = entryIterator.next();
	        previousPolicy.policy.put(entry.getKey(), entry.getValue());
	    }

	    boolean policyChanged = false; // Track whether any change has been made to the policy

	    Iterator<Map.Entry<Game, Move>> policyIterator = this.curPolicy.entrySet().iterator();
	    
	    // loop through all states in the current policy
	    while (policyIterator.hasNext()) {
	        Map.Entry<Game, Move> policyEntry = policyIterator.next();
	        Game state = policyEntry.getKey();
	        Move currentMove = policyEntry.getValue();

	        // Initialize the best value and move with the current policy values
	        double bestValue = this.policyValues.get(state);
	        Move bestMove = currentMove;

	        Iterator<Move> moveIterator = state.getPossibleMoves().iterator();
	        
	        // loop through all possible moves from the current state
	        while (moveIterator.hasNext()) {
	            Move move = moveIterator.next();
	            
	            qValue = calculateQValue(state, move); // Q value calculation
	            
	            if (qValue > bestValue) { // Update the best move and value if the new Q value is higher
	                bestValue = qValue;
	                bestMove = move;
	                policyChanged = true;
	            }
	        }
	        this.curPolicy.put(state, bestMove); // Update the policy with the best move found for the state
	    }

	    return policyChanged;
	}
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
	    initRandomPolicy(); // Policy Initialization for every state
	    boolean policyImproved = true;  // to track whether the policy improves during each iteration, set to true so that the loop starts
	    
	    // this loops until the policy no longer improves i.e is converged
	    while (policyImproved) {
	        evaluatePolicy(delta); // perform policy evaluation
	        policyImproved = improvePolicy(); // perform policy improvement
	    }
	    super.policy = new Policy(curPolicy); // assign the converged policy to the agent
	}
	
	/**
	 * calculates the Q-value for a given state and move using the transition probabilities.
	 * @param state a given state
	 * @param move a given action
	 * @return the calculated q value using bellman's eqn for Q(state,move)
	 */
	private double calculateQValue(Game state, Move move) {
	    double qValue = 0; // Value Initialization
	    
	    // getting the transitions for the given state and move
	    List<TransitionProb> transitions = mdp.generateTransitions(state, move);
	    Iterator<TransitionProb> transitionIterator = transitions.iterator();
	    
	    // Performs q value calculation for a given state with all possible moves
	    while (transitionIterator.hasNext()) {
	        TransitionProb transition = transitionIterator.next();
	        double reward = transition.outcome.localReward;
	        double futureValue = this.policyValues.get(transition.outcome.sPrime);
	        qValue += transition.prob * (reward + (discount * futureValue)); // Value Updated
	    }
	    
	    return qValue;
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}
