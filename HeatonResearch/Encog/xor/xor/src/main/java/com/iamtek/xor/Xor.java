package com.iamtek.xor;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
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
		MLDataSet data = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		ResilientPropagation train = new ResilientPropagation(network, data);
		int iteration = 0;
		while (true) {
			train.iteration();
			iteration++;
			System.out.println("#" + iteration + ", Error: " + train.getError());
			if (train.getError() < 0.01) break;
		}
		train.finishTraining();
		
		//Test
		System.out.println("Results:");
		for (MLDataPair pair : data) {
			MLData output = network.compute(pair.getInput());
			System.out.println(
					pair.getInput().getData(0) + ", " + pair.getInput().getData(1) + ", Actual=" + output.getData(0) + ", Ideal=" + pair.getIdeal().getData(0)
			);
		}
		
		Encog.getInstance().shutdown();

	}

}
