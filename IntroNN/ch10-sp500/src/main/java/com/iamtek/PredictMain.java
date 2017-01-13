package com.iamtek;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Date;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.HybridStrategy;
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
	public static int OUTPUT_SIZE = 1;
	public static int HIDDEN1 = 20;
	public static int HIDDEN2 = 0;
	public static double MAX_ERROR = 0.001;
	public static Date PREDICT_FROM = ReadCSV.parseDate("2016-01-01");
	public static Date LEARN_FROM = ReadCSV.parseDate("1980-01-01");
	public static String FILE_PATH = "src/main/resources/";
	public static String FINANCE_FILE = "sp500.txt";
	public static String RATE_FILE = "prime.txt";
	public static String DATA_SAVE_FILE = "save.txt";
	public static String NN_FILE = "NN.txt";

	public static void main(String[] args) {

		PredictMain p = new PredictMain();
		p.run(true);
		//p.run(false);
		Encog.getInstance().shutdown();
	}
	
	public void run(boolean fullmode){
			try {
				this.actual = new SP500Actual(INPUT_SIZE, OUTPUT_SIZE);
				this.actual.generateData(FILE_PATH, FINANCE_FILE, RATE_FILE);
				System.out.println("Samples read:"+this.actual.size());
				if(fullmode){
					generateTrainingSets();
					createNetwork();
					saveNetwork(FILE_PATH, NN_FILE);
					trainNetwork();
				}else{
					loadNetwork(FILE_PATH, NN_FILE);
				}
				display();
			} catch (Exception e) {
				e.printStackTrace();
			}
	}
	
	private void display() {
		// TODO Auto-generated method stub
		
	}

	private double[][] input;
	private double[][] ideal;
	private BasicNetwork network;
	private SP500Actual actual;
	
	private void generateTrainingSets(){
		this.input = new double[TRAINING_SIZE][INPUT_SIZE*2];
		this.ideal = new double[TRAINING_SIZE][OUTPUT_SIZE];
		int startIndex = 0;
		for (FinancialSample sample : this.actual.getSamples()) {	//find start point of training
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
			this.actual.getInputData(startIndex+(i*factor), this.input[i]);
			this.actual.getOutputData(startIndex+(i*factor), this.ideal[i]);
		}
	}
	
	private void createNetwork(){
		NeuralNetworkPattern pattern = new FeedForwardPattern();
		pattern.setActivationFunction(new ActivationTANH());
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
		DecimalFormat fm = new DecimalFormat("##.####");
		MLDataSet training = new BasicMLDataSet(input, ideal);
		MLTrain train = new Backpropagation(
				this.network, training,
				0.00001, 0.1);
		CalculateScore score = new TrainingSetScore(training);
		MLTrain train2 = new NeuralSimulatedAnnealing(
				this.network, score,
				10, 2, 100);
		train.addStrategy(new HybridStrategy(train2));
		do{
			train.iteration();
			System.out.println("Iteration #"+train.getIteration()+" Error:"+fm.format(train.getError()));
		} while (train.getError()>MAX_ERROR);
		train.finishTraining();
	}

}
