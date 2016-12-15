package com.iamtek.persistence;

import java.io.File;
import java.util.Arrays;

import org.encog.Encog;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;

public class EncogPersist {
	private static final String NNFile = "src/main/resources/encogpersist.eg";
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

	public void trainAndSave(){
		System.out.println("Train XOR network to under 1% error rate.");
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(2));
		network.addLayer(new BasicLayer(6));
		network.addLayer(new BasicLayer(1));
		network.getStructure().finalizeStructure();
		network.reset();
		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		MLTrain train = new ResilientPropagation(network, trainingSet);
		while(true){
			train.iteration();
			if(train.getError() < 0.01) break;
		}
		System.out.println("Network trained to error:" + network.calculateError(trainingSet));
		
		System.out.println("Saving network...");
		EncogDirectoryPersistence.saveObject(new File(NNFile), network);
	}
	
	public void loadAndEvaluate(){
		System.out.println("Loading network...");
		BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File(NNFile));
		
		MLDataSet evaluationSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		System.out.println("Loaded network error:" + network.calculateError(evaluationSet));
		
		System.out.println("Network results");
		for(MLDataPair pair : evaluationSet){
			double[] input = pair.getInputArray();
			double[] output = {0};
			double[] ideal = pair.getIdealArray();
			network.compute(input, output);
			System.out.println("Input:" + Arrays.toString(input) + " Ideal:" + Arrays.toString(ideal) + " Predicted:" + Arrays.toString(output));
		}
	}
	
	public static void main(String[] args) {
		EncogPersist persist = new EncogPersist();
		persist.trainAndSave();
		persist.loadAndEvaluate();
		
		Encog.getInstance().shutdown();
	}
	
}
