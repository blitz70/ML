package com.iamtek;

import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.neural.networks.BasicNetwork;

public class PilotScore implements CalculateScore {
	@Override
	public double calculateScore(MLMethod method) {
		BasicNetwork network = (BasicNetwork) method;
		NeuralPilot pilot = new NeuralPilot(network, false);
		return pilot.scorePilot();
	}

	@Override
	public boolean shouldMinimize() {
		return false;
	}

	@Override
	public boolean requireSingleThreaded() {
		return false;
	}

}
