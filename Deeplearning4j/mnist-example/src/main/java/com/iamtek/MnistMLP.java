package com.iamtek;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistMLP {
  
    private static Logger log = LoggerFactory.getLogger(MnistMLP.class);

	public static void main(String[] args) throws Exception {
		
		int noRows = 28; // The number of rows of a matrix.
	    int noColumns = 28; // The number of columns of a matrix.
	    int noOutputs = 10; // Number of possible outcomes (e.g. labels 0 through 9).
	    int batchSize = 128; // How many examples to fetch with each step.
	    int seed = 123; // This random-number generator applies a seed to ensure that the same initial weights are used when training. We’ll explain why this matters later.
	    int noEpochs = 15; // An epoch is a complete pass through a given dataset.
	    
	    //get data
	    log.info("Getting data...");
	    DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, seed); 
	    DataSetIterator testData = new MnistDataSetIterator(batchSize, false, seed);
	    
	    //setup neural network
        log.info("Creating network...");
	    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	    	.seed(seed)
	    	.iterations(1)
	    	.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	    	.updater(Updater.NESTEROVS).momentum(0.9).learningRate(0.006)
	    	.regularization(true).l2(1e-4)
	    	.list()
	    		.layer(0, new DenseLayer.Builder()
    				.nIn(noRows*noColumns)
    				.nOut(1000)
    				.activation(Activation.RELU)
    				.weightInit(WeightInit.XAVIER)
    				.build())
	    		.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
	    			.nIn(1000)
	    			.nOut(noOutputs)
	    			.activation(Activation.SOFTMAX)
	    			.weightInit(WeightInit.XAVIER)
	    			.build())
	    	.pretrain(false).backprop(true).build();
	    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    	model.init();
		model.setListeners(new ScoreIterationListener(10));
		
		//train
        log.info("Training...");
		for (int i = 0; i < noEpochs; i++) {
			model.fit(trainData);
		}
		
		//test
        log.info("Testing...");
		Evaluation eval = new Evaluation(noOutputs);
		while(testData.hasNext()){
			DataSet ds = testData.next();
			INDArray features = ds.getFeatures();
			INDArray labels = ds.getLabels();
			INDArray predicted = model.output(features);
			eval.eval(labels, predicted);
		}
        log.info(eval.stats());
	}
	
}
