package com.iamtek;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class MLPClassifierLinear {

	private static String FILE_PATH = "src/main/resources/";
	private static String TRAIN_FILE = "linear_data_train.csv";
	private static String EVAL_FILE = "linear_data_eval.csv";

	public static void main(String[] args) throws Exception {
		
		int seed = 123;
		double learningRate = 0.01;
		int batchSize = 50;
		int nEpochs = 30;
		
		int nInputs = 2;
		int nOutputs = 2;
		int nHiddens = 20;
		
		//load training data
		RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(new File(FILE_PATH+TRAIN_FILE)));
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);
		
		//load evaluating data
		RecordReader rr2 = new CSVRecordReader();
		rr2.initialize(new FileSplit(new File(FILE_PATH+EVAL_FILE)));
		DataSetIterator evalIter = new RecordReaderDataSetIterator(rr2, batchSize, 0, 2);

		//setup neural network
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.seed(seed)
			.iterations(1)
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
			.learningRate(learningRate)
			.updater(Updater.NESTEROVS).momentum(0.9)
			.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(nInputs)
						.nOut(nHiddens)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.RELU)
						.build())
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
						.nIn(nHiddens)
						.nOut(nOutputs)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.SOFTMAX)
						.build())
			.pretrain(false).backprop(true).build();
		//System.out.println(conf.toJson());
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
			model.init();
			model.setListeners(new ScoreIterationListener(10));
		
		//train
		System.out.println("Training model...");
		for (int i = 0; i < nEpochs; i++) {
			model.fit(trainIter);
		}
		
		//evaluate
		System.out.println("Evaluating model...");
		Evaluation eval = new Evaluation(nOutputs);
		while(evalIter.hasNext()){
			DataSet ds = evalIter.next();
			INDArray features = ds.getFeatures();
			INDArray labels = ds.getLabels();
			INDArray predicted = model.output(features, false);
			eval.eval(labels, predicted);
		}
		System.out.println(eval.stats());
	}

}
