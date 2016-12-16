package com.iamtek.train_supervised;

import org.encog.ml.MLRegression;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.CalculateScore;

public class PilotScore implements CalculateScore {
	/*@Override
	public double calculateScore(MLMethod method) {
		BasicNetwork network = (BasicNetwork) method;
		NeuralPilot pilot = new NeuralPilot(network, false);
		return pilot.scorePilot();
	}*/

	@Override
	public boolean shouldMinimize() {
		return false;
	}

	/*@Override
	public boolean requireSingleThreaded() {
		return false;
	}*/

	@Override
	public double calculateScore(MLRegression method) {
		BasicNetwork network = (BasicNetwork) method;
		NeuralPilot pilot = new NeuralPilot(network, false);
		return pilot.scorePilot();
	}

}
