package com.iamtek;

import java.text.DecimalFormat;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.neural.pattern.NeuralNetworkPattern;

public class Sinewave {

	public static int ACTUAL_SIZE = 500;
	public static int TRAINING_SIZE = 250;
	public static int INPUT_SIZE = 5;
	public static int OUTPUT_SIZE = 1;
	public static int HIDDEN1 = 7;
	public static int HIDDEN2 = 3;
	public static boolean USE_BACKPROP = true;
	
	private ActualData actual;
	private double[][] input;
	private double[][] ideal;
	private BasicNetwork network;

	public static void main(String[] args) {
		Sinewave wave = new Sinewave();
		wave.createNetwork();
		wave.generateData();
		wave.train();
		wave.test();
		Encog.getInstance().shutdown();
	}
	
	private void createNetwork(){
		NeuralNetworkPattern pattern = new FeedForwardPattern();
		pattern.setActivationFunction(new ActivationTANH());
		pattern.setInputNeurons(INPUT_SIZE);
		pattern.setOutputNeurons(OUTPUT_SIZE);
		if(HIDDEN1>0){
			pattern.addHiddenLayer(HIDDEN1);
		}
		if(HIDDEN2>0){
			pattern.addHiddenLayer(HIDDEN2);
		}
		network = (BasicNetwork) pattern.generate();
		network.reset();
	}

	private void generateData() {
		this.actual = new ActualData(ACTUAL_SIZE, INPUT_SIZE, OUTPUT_SIZE);
		this.input = new double[TRAINING_SIZE][INPUT_SIZE];
		this.ideal = new double[TRAINING_SIZE][OUTPUT_SIZE];
		for (int i = 0; i < TRAINING_SIZE; i++) {
			this.input[i] = this.actual.getInputData(i);
			this.ideal[i] = this.actual.getOutputData(i);
		}
	}
	
	private void train() {
		MLTrain train;
		int iteration = 1;
		if(USE_BACKPROP) {
			train = new Backpropagation(
					network, new BasicMLDataSet(input, ideal),
					0.001, 0.1);
		}
		else {
			train = new NeuralSimulatedAnnealing(
					network, new TrainingSetScore(new BasicMLDataSet(input, ideal)),
					10, 2, 100);
		}
		while (true){
			train.iteration();
			System.out.println("Iteration #"+iteration+", Error:"+train.getError());
			iteration++;
			if((train.getError() < 0.001) || (iteration > 5000)) break;
		}
	}
	
	public void test(){
		DecimalFormat fm = new DecimalFormat("#.###");
		for (int i = INPUT_SIZE; i < ACTUAL_SIZE-(OUTPUT_SIZE-1); i++) {
			MLData in = new BasicMLData(this.actual.getInputData(i-INPUT_SIZE));
			MLData out = new BasicMLData(this.actual.getOutputData(i-INPUT_SIZE));
			MLData predict = network.compute(in);
			StringBuilder str = new StringBuilder();
			str.append("#"+i);
			str.append(" Actual[");
			for (int j = 0; j < out.size(); j++) {
				if (j > 0) {
					str.append(',');
				}
				str.append(fm.format(out.getData(j)));
			}
			str.append("]  Predict[");
			for (int j = 0; j < predict.size(); j++) {
				if (j > 0) {
					str.append(',');
				}
				str.append(fm.format(predict.getData(j)));
			}
			ErrorCalculation error = new ErrorCalculation();
			error.updateError(predict.getData(), out.getData(), 1);
			str.append("] Error["+fm.format(error.calculateRMS()*100)+"%]");
			System.out.println(str.toString());
		}
	}

}
