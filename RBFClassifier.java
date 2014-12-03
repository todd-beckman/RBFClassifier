package RBFClassifier;

import java.util.HashMap;

public class RBFClassifier {
	
	protected RBFNetwork network;
	
	/**
	 * Builds a manager for a Radio-Basis Function network which will classify
	 * input into num_output possible classes.
	 * @param num_inputs The number of inputs into the network
	 * @param num_gaussian The number of hidden nodes in the network
	 * @param num_output The number of categories (the output layer size).
	 */
	public RBFClassifier(int num_inputs, int num_gaussian, int num_output, float learning_rate, float gaussian_width) {
		network = new RBFNetwork(num_inputs, num_gaussian, num_output, learning_rate, gaussian_width);
	}
	
	/**
	 * Classifies (tests) the input
	 * @param input The state
	 * @return The class fitting that input
	 */
	public int classify(float[] input) {
		
		//	First read from the the RBF network
		float[] class_vals = network.get_output(input);
		
		//	Find the argmax value (the most likely category)
		int best = 0;
		for (int i = 0; i < class_vals.length; i++) {
			if (class_vals[best] < class_vals[i]) {
				best = i;
				
			}
		}
		
		return best;
	}
	
	
	public void learn(float[] input, float[] outcome) {
		float[] out = network.get_output(input);
		
		
		//	Learning requires "backpropagation"
		//	swapping these parameters fixed the world somehow.
		network.back_propogate(
				
				//	The expected is what the network thinks will happen
				outcome,
				
				//	The actual is given based on the data
				out);
	}

	/**
	 * Sets the weights of the hidden layer based on the data. Also writes
	 * the data to a file to read for later.
	 * @param filename The output file
	 * @param data The experimental data
	 * @param num_gaussian
	 */
	public void set_weights_w(String filename, float[][] data, int num_gaussian) {
		
		//	make k-means clusters
		float[][] kmeans = kmeans(data, num_gaussian);
		
		Main.float_to_csv(filename, kmeans);

		//	Simply these clusters to the hidden layer
		for (int i = 0; i < kmeans.length; i++) {
			network.gnodes[i].set_centers(kmeans[i]);
		}
	}
	
	/**
	 * Sets the weights of the hidden layer based on the data. Also writes
	 * the data to a file to read for later.
	 * @param filename The input file
	 */
	public void set_weights_r(String filename) {
		float[][] centers = Main.csv_to_float("centers.csv");
		for (int i = 0; i < centers.length; i++) {
			network.gnodes[i].set_centers(centers[i]);
		}
	}
	

	/**
	 * Applies k-means clustering on a data file into a given number of
	 * clusters. This will generate locally optimal cluster centers in
	 * a finite number of iterations.
	 * @param data The input data
	 * @param num_gaussian The number of clusters
	 * @return A list of centers for each hidden Gaussian node
	 */
    private static float[][] kmeans(float[][] data, int num_gaussian) {
    	
    	//	Make the list of centers- num_gaussian lists, each at the
    	//	length of the network input
        float[][] centers = new float[num_gaussian][data[0].length];
        
        //	Initalize the first centers- force the initial clusters
        System.arraycopy(data, 0, centers, 0, centers.length);
        
        //	Prepare the loop
        HashMap<float[], Integer> old_assignments = null;
        boolean changed = true;
        
        //	Run the loop. Terminate when there is no change because
        //	this function is proven to terminate without requiring
        //	an infinite convergence.
        while (changed) {
        	
            changed = false;
            HashMap<float[], Integer> assignments = new HashMap<>();
            float[][] new_centers = new float[num_gaussian][data[0].length];
            
            //	Assignments
            int[] center_count = new int[num_gaussian];
            for (float[] f : data) {
            	
            	//	Find the closest center
                int min_index = 0;
                double min_dis = euclidean_distance(centers[0], f);
                for (int i = 1; i < centers.length; i++) {
                    double distance = euclidean_distance(centers[i], f);
                    if (distance < min_dis) {
                        min_dis = distance;
                        min_index = i;
                    }
                }
                
                //	Save the this data point's closest center
                assignments.put(f, min_index);
                
                //	Did the point change closest cluster since last time?
                if (old_assignments == null || old_assignments.get(f) != min_index) {
                    changed = true;
                }
                
                //	Make up new centers
                for (int i = 0; i < f.length; i++) {
                    new_centers[min_index][i] += f[i];
                }
                
                //	Count the number in this center (good for average later)
                center_count[min_index]++;
                
            }
            
            //	Updates
            for (int i = 0; i < num_gaussian; i++) {
                for (int j = 0; j < new_centers[i].length; j++) {
                	
                	//	Average the centers
                    if (center_count[i] != 0) {
                        new_centers[i][j] /= center_count[i];
                    }
                }
            }
            
            //	store this iteration for next time
            old_assignments = assignments;
            centers = new_centers;
        }
        
        //	Return the best
        return centers;
    }
    
    /**
     * Gets the Euclidean distance between two vectors
     * @param a One vector
     * @param b Another vector
     * @return The Euclidean distance
     */
    public static double euclidean_distance(float[] a, float[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i]-b[i],2);
        }
        return Math.sqrt(sum);
    }
}

