package com.iamtek.train_propagation;

import org.encog.ml.MLMethod;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.factory.MLTrainFactory;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;

public class Factories {
	
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

		//create NN
		/*BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, 2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 4));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		network.getStructure().finalizeStructure();
		network.reset();*/
		MLMethodFactory methodFactory = new MLMethodFactory();
		MLMethod method = methodFactory.create(
				MLMethodFactory.TYPE_FEEDFORWARD,
				"?:B->SIGMOID->4:B->SIGMOID->?",
				2, 1);
		
		//select training
		MLTrainFactory trainFactory = new MLTrainFactory();
		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		MLTrain train = trainFactory.create(
				method,
				trainingSet,
				MLTrainFactory.TYPE_BACKPROP,
				"LR=0.7, MOM=0.3");
		
		//train
		int epoch = 1;
		while (true) {
			train.iteration();
			System.out.println("Epoch #" + epoch + ", Error:" + train.getError());
			epoch++;
			if (train.getError() < 0.001) break;
		}
		train.finishTraining();
		
		//test
		System.out.println("Neural Network Results:");
		for (MLDataPair pair : trainingSet) {
			MLData output = ((BasicNetwork)method).compute(pair.getInput());  
			System.out.println(
					pair.getInput().getData(0) + ", " + pair.getInput().getData(1) + ", Actual=" + output.getData(0) + ",\tIdeal=" + pair.getIdeal().getData(0)
			);
		}

	}

}
