package com.iamtek;

import java.io.File;
import java.io.IOException;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Date;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.HybridStrategy;
import org.encog.ml.train.strategy.end.EndMaxErrorStrategy;
import org.encog.ml.train.strategy.end.EndMinutesStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.neural.pattern.NeuralNetworkPattern;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.csv.ReadCSV;

public class PredictMain {
	
	public static int TRAINING_SIZE = 500;
	public static int INPUT_SIZE = 10;
	public static int OUTPUT_SIZE = 1;//1
	public static int HIDDEN1 = 20;
	public static int HIDDEN2 = 0;
	public static double MAX_ERROR = 0.00001;//0.00001
	public static Date PREDICT_FROM = ReadCSV.parseDate("2016-01-01");
	public static Date LEARN_FROM = ReadCSV.parseDate("1980-01-01");
	public static String FILE_PATH = "src/main/resources/";
	public static String FINANCE_FILE = "sp500.txt";
	public static String RATE_FILE = "prime.txt";
	public static String DATA_FILE = "data.txt";
	public static String NN_FILE = "NN.txt";
	
	private double[][] input;
	private double[][] ideal;
	private BasicNetwork network;
	private SP500Data data;
	
	public static void main(String[] args) {

		PredictMain p = new PredictMain();
		//p.test();
		//p.run(true);
		p.run(false);
		Encog.getInstance().shutdown();
	}
	
	public void test(){
		try {
			this.data = new SP500Data(INPUT_SIZE, OUTPUT_SIZE);
			this.data.generateData(FILE_PATH, FINANCE_FILE, RATE_FILE);
			generateTrainingSets();
			System.out.println(Arrays.toString(this.input[0]));
			System.out.println(Arrays.toString(this.ideal[0]));
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	public void run(boolean fullmode){
			try {
				this.data = new SP500Data(INPUT_SIZE, OUTPUT_SIZE);
				this.data.generateData(FILE_PATH, FINANCE_FILE, RATE_FILE);
				System.out.println("Samples read:"+this.data.size());
				if(fullmode){
					generateTrainingSets();
					this.data.saveData(FILE_PATH, DATA_FILE);
					createNetwork();
					trainNetwork();
					saveNetwork(FILE_PATH, NN_FILE);
				}else{
					loadNetwork(FILE_PATH, NN_FILE);
				}
				predict();
			} catch (Exception e) {
				e.printStackTrace();
			}
	}
	
	private void predict() {
		NumberFormat fm = NumberFormat.getPercentInstance();
		fm.setMinimumFractionDigits(2);
		
		double[] present = new double[INPUT_SIZE*2];
		double[] predict = new double[OUTPUT_SIZE];
		double[] ideal = new double[OUTPUT_SIZE];
		
		int index = 0;
		double errorSum = 0;
		double count = 0;
		double hitSum = 0;
		for (FinancialSample sample : this.data.getSamples()) {
			if(sample.getDate().after(PREDICT_FROM)){
				this.data.getInputData(index-INPUT_SIZE, present);
				this.data.getOutputData(index-INPUT_SIZE, ideal);
				this.network.compute(present, predict);
				ErrorCalculation error = new ErrorCalculation();
				error.updateError(predict, ideal, 1);
				double err = error.calculateRMS();
				StringBuilder result = new StringBuilder();
				result.append(ReadCSV.displayDate(sample.getDate()));
				result.append(":Start=");
				result.append(sample.getAmount());
				result.append(", Ideal=");
				result.append(fm.format(ideal[0]));
				result.append(", Predicted=");
				result.append(fm.format(predict[0]));
				result.append(", Difference=");
				result.append(fm.format(err));
				System.out.println(result.toString());
				errorSum += err;
				count++;
				if(Math.signum(ideal[0]) == Math.signum(predict[0])){
					hitSum++;
				}
			}
			index++;
		}
		System.out.println(
				"Ave Difference:"+fm.format(errorSum/count)+
				", Hit rate:"+fm.format(hitSum/count));
	}

	private void generateTrainingSets(){
		this.input = new double[TRAINING_SIZE][INPUT_SIZE*2];
		this.ideal = new double[TRAINING_SIZE][OUTPUT_SIZE];
		int startIndex = 0;
		for (FinancialSample sample : this.data.getSamples()) {	//find start point of training
			if(sample.getDate().after(LEARN_FROM)){
				break;
			}
			startIndex++;
		}
		int actualSamples = TRAINING_SIZE - startIndex;
		if(actualSamples == 0) {
			System.out.println("Nothing to train!");
			System.exit(0);
		}
		int factor = actualSamples/TRAINING_SIZE;	//?
		for (int i = 0; i < TRAINING_SIZE; i++) {
			this.data.getInputData(startIndex+(i*factor), this.input[i]);
			this.data.getOutputData(startIndex+(i*factor), this.ideal[i]);
		}
	}
	
	private void createNetwork(){
		NeuralNetworkPattern pattern = new FeedForwardPattern();
		pattern.setActivationFunction(new ActivationTANH());//
		pattern.setInputNeurons(INPUT_SIZE*2);
		pattern.addHiddenLayer(HIDDEN1);
		if(HIDDEN2>0){
			pattern.addHiddenLayer(HIDDEN2);
		}
		pattern.setOutputNeurons(OUTPUT_SIZE);
		this.network = (BasicNetwork) pattern.generate();
		this.network.reset();
	}
	
	private void saveNetwork(String filePath, String fileName) throws IOException{
		EncogDirectoryPersistence.saveObject(new File(filePath, fileName), this.network);
		//SerializeObject.save(new File(filePath,fileName), this.network);
	}
	
	private void loadNetwork(String filePath, String fileName) throws IOException, ClassNotFoundException {
		this.network = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File(filePath,fileName));
		//this.network = (BasicNetwork) SerializeObject.load(new File(filePath,fileName));
	}
	private void trainNetwork(){
		NumberFormat fm = NumberFormat.getPercentInstance();
		fm.setMinimumFractionDigits(4);
		MLDataSet training = new BasicMLDataSet(input, ideal);
		MLTrain train = new Backpropagation(
				this.network, training,
				0.00001, 0.1);
		CalculateScore score = new TrainingSetScore(training);
		MLTrain train2 = new NeuralSimulatedAnnealing(
				this.network, score,
				10, 2, 100);
		train.addStrategy(new HybridStrategy(train2));
		train.addStrategy(new EndMaxErrorStrategy(MAX_ERROR));
		train.addStrategy(new EndMinutesStrategy(2));
		//train.addStrategy(new ResetStrategy(MAX_ERROR*10, 100));
		while(!train.isTrainingDone()){
			train.iteration();
			System.out.println("Iteration #"+train.getIteration()+" Error:"+fm.format(train.getError()));
		}
		train.finishTraining();
		System.out.println("Training done.");
	}

}
