package com.iamtek;

import org.encog.Encog;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.NEATUtil;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.util.simple.EncogUtility;

public class NEATMain {

	public static double XOR_INPUT[][] = {
			{0, 0},
			{1, 0},
			{0, 1},
			{1, 1}
	};

	public static double XOR_IDEAL[][] = {
			{0},
			{1},
			{1},
			{0}
	};

	public static void main(String[] args) {

		//neural network
		MLMethod method = new NEATPopulation(2, 1, 1000);
		((NEATPopulation) method).setInitialConnectionDensity(1.0);
		((NEATPopulation) method).reset();
		
		//train
		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		CalculateScore score = new TrainingSetScore(trainingSet);
		MLTrain train = NEATUtil.constructNEATTrainer((NEATPopulation) method, score);
		while(true){
			train.iteration();
			System.out.println("Epoch #" + train.getIteration() + ", Error:" + train.getError() + ", Species:" + ((NEATPopulation)method).getSpecies().size());
			if(train.getError() < 0.01) break;
		}
		train.finishTraining();
		NEATPopulation network = (NEATPopulation) train.getMethod();
		//((NEATPopulation) network).clearContext();
		network.reset();

		System.out.println("Neural Network Results:");
		for (MLDataPair pair : trainingSet) {
			MLData output = network.compute(pair.getInput());
			System.out.println(
					pair.getInput().getData(0) + ", " + pair.getInput().getData(1) + ", Actual=" + output.getData(0) + ",\tIdeal=" + pair.getIdeal().getData(0)
			);
		}
		EncogUtility.evaluate(network, trainingSet);
		Encog.getInstance().shutdown();

	}

}
