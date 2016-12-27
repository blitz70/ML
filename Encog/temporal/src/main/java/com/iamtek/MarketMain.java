package com.iamtek;

import java.io.File;
import java.text.DecimalFormat;
import java.util.Calendar;
import java.util.GregorianCalendar;

import org.encog.ConsoleStatusReportable;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.market.MarketDataDescription;
import org.encog.ml.data.market.MarketDataType;
import org.encog.ml.data.market.MarketMLDataSet;
import org.encog.ml.data.market.TickerSymbol;
import org.encog.ml.data.market.loader.MarketLoader;
import org.encog.ml.data.market.loader.YahooFinanceLoader;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.neural.pattern.NeuralNetworkPattern;
import org.encog.neural.prune.PruneIncremental;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.simple.EncogUtility;

enum Direction {
	up, down;
}

public class MarketMain {
	
	public static final String DIR = "src/main/resources/";
	public static final String NETWORK_FILE = "neuralNetwork.eg";
	public static final String TRAINING_FILE = "trainingData.egb";
	public static final String EVALUATING_FILE = "evaluatingData.egb";
	public static final int TRAINING_MINUTES = 1;
	public static final int HIDDEN1_COUNT= 20;
	public static final int HIDDEN2_COUNT= 0;
	public static final int PAST_WINDOW = 10;
	public static final int FUTURE_WINDOW = 1;
	public static final int INPUTS = 2;
	//public static final TickerSymbol TICKER = new TickerSymbol("AAPL") ;	//Apple
	public static final TickerSymbol TICKER = new TickerSymbol("005930.KS") ;	//Samsung Electronics
	//public static final TickerSymbol TICKER = new TickerSymbol("036570.KS") ;	//NCsoft

	public static void generate(String dir, int[] startTrain, int[] startEvaluate, int[] endEvaluate){

		//get training data, [0] years [1] days offset
		MarketLoader loader = new YahooFinanceLoader();
		MarketMLDataSet market = new MarketMLDataSet(loader, PAST_WINDOW, FUTURE_WINDOW);
		MarketDataDescription desc = new MarketDataDescription(
				TICKER,
				MarketDataType.ADJUSTED_CLOSE,
				true, true);
		market.addDescription(desc);
		desc = new MarketDataDescription(
				TICKER,
				MarketDataType.OPEN,
				true, false);
		market.addDescription(desc);
		Calendar begin = new GregorianCalendar();
		Calendar end = new GregorianCalendar();
		begin.add(Calendar.YEAR, -startTrain[0]);
		begin.add(Calendar.DATE, -startTrain[1]);
		end.add(Calendar.YEAR, -startEvaluate[0]);
		end.add(Calendar.DATE, -startEvaluate[1]);
		market.load(begin.getTime(), end.getTime());
		market.generate();
		//save training data
		EncogUtility.saveEGB(new File(dir, TRAINING_FILE), market);
		
		//get evaluating data
		market = new MarketMLDataSet(loader, PAST_WINDOW, FUTURE_WINDOW);
		desc = new MarketDataDescription(
				TICKER,
				MarketDataType.ADJUSTED_CLOSE,
				true, true);
		market.addDescription(desc);	//add more?
		desc = new MarketDataDescription(
				TICKER,
				MarketDataType.OPEN,
				true, false);
		market.addDescription(desc);	//add more?
		begin = new GregorianCalendar();
		end = new GregorianCalendar();
		begin.add(Calendar.YEAR, -startEvaluate[0]);
		begin.add(Calendar.DATE, -startEvaluate[1]);
		end.add(Calendar.YEAR, -endEvaluate[0]);
		end.add(Calendar.DATE, -endEvaluate[1]);

		
		market.load(begin.getTime(), end.getTime());
		market.generate();
		//save training data
		EncogUtility.saveEGB(new File(dir, EVALUATING_FILE), market);

		//create&save n network
		BasicNetwork network =  EncogUtility.simpleFeedForward(
				market.getInputSize(),
				HIDDEN1_COUNT*INPUTS, HIDDEN2_COUNT*INPUTS,
				market.getIdealSize(),
				true);
		EncogDirectoryPersistence.saveObject(new File(dir, NETWORK_FILE), network);

	}
	
	public static void train(String dir){
		
		//load n network & training data
		File networkFile = new File(dir, NETWORK_FILE);
		File trainingFile = new File(dir, TRAINING_FILE);
		if(!networkFile.exists()){
			System.out.println("Network file error: " + networkFile.getAbsolutePath());
			return;
		} else if (!trainingFile.exists()){
			System.out.println("Data file error: " + trainingFile.getAbsolutePath());
			return;
		}
		BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(networkFile);
		MLDataSet trainingSet = EncogUtility.loadEGB2Memory(trainingFile);
		
		//train
		EncogUtility.trainConsole(network, trainingSet, TRAINING_MINUTES);
		System.out.println("Final Error: " + network.calculateError(trainingSet) + "%. Training complete, saving network...");
		
		//save n network
		EncogDirectoryPersistence.saveObject(networkFile, network);
		System.out.println("Network saved.");
		
		Encog.getInstance().shutdown();
		
	}
	
	public static void prune(String dir){
		
		//load training data
		File trainingFile = new File(dir, TRAINING_FILE);
		if (!trainingFile.exists()){
			System.out.println("Data file error: " + trainingFile.getAbsolutePath());
			return;
		}
		MLDataSet trainingSet = EncogUtility.loadEGB2Memory(trainingFile);
		
		//create n network
		NeuralNetworkPattern pattern = new FeedForwardPattern();
		pattern.setInputNeurons(trainingSet.getInputSize());
		pattern.setOutputNeurons(trainingSet.getIdealSize());
		pattern.setActivationFunction(new ActivationTANH());
		
		//prune n network
		PruneIncremental prune = new PruneIncremental(trainingSet, pattern, 100, 1, 10, new ConsoleStatusReportable());
		prune.addHiddenLayer(1, 50*INPUTS);
		prune.addHiddenLayer(0, 50*INPUTS);
		prune.process();

		//save best n network
		File networkFile = new File(dir, NETWORK_FILE);
		EncogDirectoryPersistence.saveObject(networkFile, prune.getBestNetwork());
		System.out.println("Best network saved.");

	}

	public static Direction determineDirection(double d){
		if(d>0){
			return Direction.up;
		} else return Direction.down;
		
	}
	
	public static void evaluate(String dir){
		
		//load n network
		File networkFile = new File(dir, NETWORK_FILE);
		File evaluatingFile = new File(dir, EVALUATING_FILE);
		if(!networkFile.exists()){
			System.out.println("Network file error: " + networkFile.getAbsolutePath());
			return;
		}
		BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(networkFile);
		
		//load evaluating data
		if(!evaluatingFile.exists()){
			System.out.println("Data  file error: " + evaluatingFile.getAbsolutePath());
			return;
		}
		MLDataSet evaluatingSet = EncogUtility.loadEGB2Memory(evaluatingFile);
		DecimalFormat df = new DecimalFormat("#0.0000");
		int count=0, correct=0;
		double diffSum=0;
		for(MLDataPair pair : evaluatingSet){
			MLData input = pair.getInput();
			MLData ideal = pair.getIdeal();
			MLData output = network.compute(input);
			double actual = ideal.getData(0);
			double predict = output.getData(0);
			double diff = Math.abs(predict-actual);
			Direction actualDirection = determineDirection(actual);
			Direction predictDirection = determineDirection(predict);
			if (actualDirection == predictDirection) correct++;
			count++;
			diffSum+=diff;
			System.out.println("Day " + count +
					", Actual=" + df.format(actual) + "(" + actualDirection + ")" +
					", Predicted=" + df.format(predict) + "(" + predictDirection + ")" +
					", diff=" + diff);
		}
		double percent = (double)correct/(double)count;
		double diffAve = diffSum/(double)count;
		System.out.println("Correct/Count:"+correct+"/"+count+", Direction match: " + df.format(percent*100)+"%, Ave diff:"+df.format(diffAve));

	}
	
	public static void main(String[] args) {
		
		//generate(DIR, new int[]{2, 60}, new int[]{0, 60}, new int[]{0, 0});
		//generate(DIR, new int[]{11, 0}, new int[]{1, 0}, new int[]{0, 0});
		
		prune(DIR);
		//train(DIR);
		evaluate(DIR);

	}

}
