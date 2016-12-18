package com.iamtek;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class Xor {

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
		
		//Create
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, 2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		network.getStructure().finalizeStructure();
		network.reset();
		
		//Train
		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		MLTrain train = new ResilientPropagation(network, trainingSet);
		
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
