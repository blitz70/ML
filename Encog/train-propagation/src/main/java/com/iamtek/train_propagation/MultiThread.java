package com.iamtek.train_propagation;

import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.benchmark.RandomTrainingFactory;

public class MultiThread {

	public static final int INPUT = 40;
	public static final int HIDDEN = 60;
	public static final int OUTPUT = 20;

	public static BasicNetwork createNet(){
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(INPUT));
		network.addLayer(new BasicLayer(HIDDEN));
		network.addLayer(new BasicLayer(OUTPUT));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}
	
	public static double trainAndEvaluate(int threads, BasicNetwork network, MLDataSet trainingSet){
		MLTrain train = new ResilientPropagation(network, trainingSet);
		
		((ResilientPropagation)train).setThreadCount(threads);
		double start = System.currentTimeMillis();
		for (int i = 0; i <= 20; i++) {
			train.iteration();
			System.out.println("Iteration #" + i + ", Error:" + train.getError());
		}
		train.finishTraining();
		double time = (System.currentTimeMillis() - start)/1000;
		System.out.println("Error:"+network.calculateError(trainingSet)+" Time:"+time+"s");
		return time;
	}
	
	public static void main(String[] args) {
		BasicNetwork network = createNet();
		MLDataSet trainingSet = RandomTrainingFactory.generate(1000, 50000, INPUT, OUTPUT, -1, 1);
		double single = trainAndEvaluate(1, network, trainingSet);
		double multi = trainAndEvaluate(0, network, trainingSet);
		System.out.println("Improvement:" + single/multi*100 + "%");
	}

}
