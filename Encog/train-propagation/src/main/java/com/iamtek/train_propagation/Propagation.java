package com.iamtek.train_propagation;

import org.encog.Encog;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.lma.LevenbergMarquardtTraining;

public class Propagation {
	
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

	public static BasicNetwork createNet() {
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(2));
		network.addLayer(new BasicLayer(3));
		network.addLayer(new BasicLayer(1));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}
	
	public static void trainNet(MLTrain train){
		int epoch = 1;
		long start = System.currentTimeMillis();
		while (true) {
			train.iteration();
			System.out.println("Epoch #" + epoch + ", Error:" + train.getError());
			epoch++;
			if (train.getError() < 0.001) break;
		}
		train.finishTraining();
		System.out.println("Time:"+(System.currentTimeMillis()-start));
	}
	
	public static void testNet(BasicNetwork network, MLDataSet trainingSet){
		System.out.println("Neural Network Results:");
		for (MLDataPair pair : trainingSet) {
			MLData output = network.compute(pair.getInput());
			System.out.println(
					pair.getInput().getData(0) + ", " + pair.getInput().getData(1) + ", Actual=" + output.getData(0) + ",\tIdeal=" + pair.getIdeal().getData(0)
			);
		}
	}
	
	public static void main(String[] args) {
		
		BasicNetwork network = createNet();
		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

		//trainNet(new Backpropagation(network, trainingSet, 0.7, 0.3));
		
		//trainNet(new ManhattanPropagation(network, trainingSet, 0.00001));
		
		//trainNet(new QuickPropagation(network, trainingSet, 1.5));
		
		/*ResilientPropagation train = new ResilientPropagation(network, trainingSet);
		train.setRPROPType(RPROPType.iRPROPp);
		trainNet(train);*/
				
		//trainNet(new ScaledConjugateGradient(network, trainingSet));
		
		trainNet(new LevenbergMarquardtTraining(network, trainingSet));	//Fast!
		
		testNet(network, trainingSet);
		
		Encog.getInstance().shutdown();

	}

}
