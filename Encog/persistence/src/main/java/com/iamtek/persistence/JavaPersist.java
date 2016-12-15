package com.iamtek.persistence;

import java.io.File;
import java.io.IOException;
import java.math.RoundingMode;
import java.text.DecimalFormat;

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
import org.encog.util.obj.SerializeObject;

public class JavaPersist {
	private static final String NNFile = "src/main/resources/javapersist.ser";
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
	public static DecimalFormat df = new DecimalFormat("#.###");

	public void trainAndSave() {
		System.out.println("Train XOR network to under 1% error rate.");
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, 2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 6));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		network.getStructure().finalizeStructure();
		network.reset();
		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		MLTrain train = new ResilientPropagation(network, trainingSet);
		while(true){
			train.iteration();
			if(train.getError() < 0.01) break;
		}
		System.out.println("Network trained to error:" + df.format(network.calculateError(trainingSet)));
		
		System.out.println("Saving network...");
		try {
			SerializeObject.save(new File(NNFile), network);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void loadAndEvaluate() {
		System.out.println("Loading network...");
		try {
			BasicNetwork network = (BasicNetwork) SerializeObject.load(new File(NNFile));
			MLDataSet evaluationSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
			System.out.println("Loaded network error:" + df.format(network.calculateError(evaluationSet)));
			System.out.println("Network results");
			for (MLDataPair pair : evaluationSet) {
				MLData output = network.compute(pair.getInput());
				System.out.println(
						pair.getInput().getData(0) + ", " + pair.getInput().getData(1) + ", Actual=" + df.format(output.getData(0)) + ",\tIdeal=" + pair.getIdeal().getData(0)
				);
			}
		} catch (ClassNotFoundException | IOException e) {
			e.printStackTrace();
		}

	}

		
	public static void main(String[] args) {

		df.setRoundingMode(RoundingMode.CEILING);

		JavaPersist persist = new JavaPersist();
		persist.trainAndSave();
		persist.loadAndEvaluate();
		
		Encog.getInstance().shutdown();
	}

}
