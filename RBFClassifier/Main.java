package RBFClassifier;

import java.io.*;

public class Main {
	public static void main(String[] args) {
		int num_input = 2;
		int num_gaussian = 50;
		int num_output = 4;
		float learning_rate = 0.1f;
		float gaussian_width = 0.02f;
		
		RBFClassifier classifier = new RBFClassifier(num_input, num_gaussian, num_output, learning_rate, gaussian_width);
		
		int training_size = 100;
		
		float[][] data = new float[training_size][];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new float[2];
			data[i][0] = (float)(Math.random()) * 2f - 1f;
			data[i][1] = (float)(Math.random()) * 2f - 1f;
		}
		
		classifier.set_weights_w("centers.csv", data, num_gaussian);

		//	 Learn to classify
		for (int i = 0; i < 100000; i++) {
			teach_weight(classifier, true);
		}
		
		//	Test the weights
		for (int i = 0; i < 16; i++) {
			teach_weight(classifier, false);
		}
		
		
	}
	
	private static void teach_weight(RBFClassifier classifier, boolean learn){
		float[] sample = {(float)(Math.random()) * 2f - 1f, (float)(Math.random()) * 2f - 1f};
		
		float[] actual = {0f, 0f, 0f, 0f};
		
		int right = 0;
		
		if (sample[0] < 0) {
			if (sample[1] < 0) {
				right = 0;
			}
			else {
				right = 1;
			}
		}
		else {
			if (sample[1] < 0) {
				right = 2;
			}
			else {
				right = 3;
			}
		}
		actual[right] = 1f;
		
		if (learn) {
			classifier.learn(sample, actual);
		}
		else {
			int output = classifier.classify(sample);
			
			System.out.println("Classified (" + sample[0] + ", " + sample[1] + ") as " + output + ". Actual was " + right);
		}
	}
	
	public static float[][] csv_to_float(String filename) {
		BufferedReader file = null;
		String raw = "";
		try {
			file = new BufferedReader(new FileReader(filename));
			while (file.ready()) {
				raw += file.readLine() + "\n";
			}
			file.close();
		}
		catch (IOException e) {
			return null;
		}
		
		String[] lines = raw.split("\n");
		float[][] data = new float[lines.length][];
		for (int i = 0; i < lines.length; i++) {
			String[] line = lines[i].split(",");
			data[i] = new float[line.length];
			for (int j = 0; j < line.length; j++) {
				data[i][j] = Float.parseFloat(line[j]);
			}
		}
		
		return data;
	}
	
	public static void float_to_csv(String filename, float[][] data) {
		BufferedWriter writer = null;
		try {
			writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(filename), "utf-8"));
			
			//	Clear the file
			writer.write("");
			for (int i = 0; i < data.length; i++){
				
				//	Dirty implode (Java so lame)
				String line = "";
				for (int j = 0; j < data[i].length; j++) {
					line += data[i][j] + ",";
				}
				writer.append(line.substring(0, line.length() - 1));
				if (i < data.length - 1) {
					writer.append("\n");
				}
			}
		}
		catch (IOException ex) {} 
		finally {
			try {
				writer.close();
			} 
			catch (Exception ex) {}
		}
	}
}

