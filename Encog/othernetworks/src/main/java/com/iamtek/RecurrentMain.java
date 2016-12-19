package com.iamtek;

import org.encog.Encog;
import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.Greedy;
import org.encog.ml.train.strategy.HybridStrategy;
import org.encog.ml.train.strategy.StopTrainingStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

public class RecurrentMain {
	
	public static void main(String[] args) {
		MLDataSet trainingSet = new TemporalXor().generate(120);
		BasicNetwork ff = NNet.createFFNet();
		BasicNetwork elman = NNet.createElmanNet();
		BasicNetwork jordan = NNet.createJordanNet();
		double ffError = trainNet("Feedforward", ff, trainingSet);		//0.25
		double elmanError = trainNet("Elman", elman, trainingSet);		//0.01
		double jordanError = trainNet("Jordam", jordan, trainingSet);	//0.25 because of only 1 output
		System.out.println("Feedforward best error rate: " + ffError);
		System.out.println("Elman best error rate: " + elmanError);
		System.out.println("Jordan best error rate: " + jordanError);
		Encog.getInstance().shutdown();
	}
	
	public static double trainNet(String type, BasicNetwork network, MLDataSet trainingSet){
		//hybrid training, main Backpropagation, sub Simulated Annealing
		CalculateScore score = new TrainingSetScore(trainingSet);
		MLTrain trainMain = new Backpropagation(network, trainingSet, 0.000001, 0.0);
		MLTrain trainSub = new NeuralSimulatedAnnealing(network, score, 10, 2, 100);
		//train strategies, greedy + hybrid + stop
		StopTrainingStrategy stop = new StopTrainingStrategy();
		trainMain.addStrategy(new Greedy());
		trainMain.addStrategy(new HybridStrategy(trainSub));
		trainMain.addStrategy(stop);
		//train
		int epoch = 1;
		while (!stop.shouldStop()) {
			trainMain.iteration();
			System.out.println("Training " + type + ", Epoch #" + epoch + ", Error:" + trainMain.getError());
			epoch++;
		}
		trainMain.finishTraining();
		return trainMain.getError();
	}

}