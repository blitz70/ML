package com.iamtek;

import java.util.Arrays;

import org.encog.Encog;
import org.encog.ml.data.MLData;
import org.encog.ml.data.specific.BiPolarNeuralData;
import org.encog.neural.art.ART1;

public class ART1Main {
	public static int INPUT_NEURONS = 5;
	public static int OUTPUT_NEURONS = 10;
	public static String[] PATTERN = {
			"   O ",
            "  O O",
            "    O",
            "  O O",
            "    O",
            "  O O",
            "    O",
            " OO O",
            " OO  ",
            " OO O",
            " OO  ",
            "OOO  ",
            "OO   ",
            "O    ",
            "OO   ",
            "OOO  ",
            "OOOO ",
            "OOOOO",
            "O    ",
            " O   ",
            "  O  ",
            "   O ",
            "    O",
            "  O O",
            " OO O",
            " OO  ",
            "OOO  ",
            "OO   ",
            "OOOO ",
            "OOOOO"
	};
	
	private boolean[][] input;
	
	public void setupInput(){
		this.input = new boolean[PATTERN.length][INPUT_NEURONS];
		for (int i = 0; i < PATTERN.length; i++) {
			for (int j = 0; j < INPUT_NEURONS; j++) {
				input[i][j] = (PATTERN[i].charAt(j) == 'O');
			}
		}
		System.out.println(Arrays.deepToString(PATTERN));
		System.out.println(Arrays.deepToString(input));
	}
	
	public void run(){
		ART1 network = NNet.createART1Net(INPUT_NEURONS, OUTPUT_NEURONS);
		for (int i = 0; i < PATTERN.length; i++) {
			MLData in = new BiPolarNeuralData(input[i]);
			MLData out = new BiPolarNeuralData(OUTPUT_NEURONS);
			network.compute((BiPolarNeuralData)in, (BiPolarNeuralData)out);
			if (network.hasWinner()) {
				System.out.println(PATTERN[i] + " - " + network.getWinner());
			}
			else {
				System.out.println(PATTERN[i] + " - new Input and all Classes exhausted");
			}
		}
	}

	public static void main(String[] args) {
		ART1Main art = new ART1Main();
		art.setupInput();
		art.run();
		Encog.getInstance().shutdown();
	}

}
