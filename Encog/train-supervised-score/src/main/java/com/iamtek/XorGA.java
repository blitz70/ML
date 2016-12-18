package com.iamtek;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.MethodFactory;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;

public class XorGA {

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

	public static BasicNetwork createNet(){
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, 2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 5));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}
	
	public static void main(String[] args) {
		
		//Create
		BasicNetwork network = createNet();
		
		//Train
		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		
		//Genetic Algorithm
		CalculateScore score = new TrainingSetScore(trainingSet);
		MLTrain train = new MLMethodGeneticAlgorithm(new MethodFactory() {
			@Override
			public MLMethod factor() {
				return createNet();
			}
		} , score, 5000);
		
		int epoch = 1;
		while (true) {
			train.iteration();
			System.out.println("Epoch #" + epoch + ", Error:" + train.getError());
			epoch++;
			if (train.getError() < 0.001) break;
		}
		train.finishTraining();
		
		//Test
		System.out.println("Neural Network Results:");
		for (MLDataPair pair : trainingSet) {
			MLData output = network.compute(pair.getInput());
			System.out.println(
					pair.getInput().getData(0) + ", " + pair.getInput().getData(1) + ", Actual=" + output.getData(0) + ",\tIdeal=" + pair.getIdeal().getData(0)
			);
		}
		
		Encog.getInstance().shutdown();

	}

}
