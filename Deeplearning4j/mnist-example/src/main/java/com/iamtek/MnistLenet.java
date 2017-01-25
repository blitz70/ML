package com.iamtek;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistLenet {

  private static Logger log = LoggerFactory.getLogger(MnistLenet.class);
  
  public static void main(String[] args) throws Exception {

    int nChannels = 1;  //number of input channels
    int nOutput = 10;
    int batchSize = 64;
    int seed = 123;
    int nEpochs = 1;
    
    //create an iterator of batch size for one iteration
    log.info("Getting data...");
    DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, seed);
    DataSetIterator testData = new MnistDataSetIterator(batchSize, false, seed);
    
    //create neural network
    log.info("Creating network...");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .iterations(nEpochs)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(Updater.NESTEROVS)
        .regularization(true).l2(0.0005)
        .learningRate(0.01)
        //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
        .weightInit(WeightInit.XAVIER)
        .list()
        .layer(0, new ConvolutionLayer.Builder(5, 5)
            //specify depth, nIn is nChannels nOut is number of filters to be applied
            .nIn(nChannels)
            .nOut(20)
            .activation(Activation.IDENTITY)
            .stride(1,1)
            .build())
        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
            .kernelSize(2,2)
            .stride(2,2)
            .build())
        .layer(2, new ConvolutionLayer.Builder(5, 5)
            //no need to set nIn for later layers
            .nOut(50)
            .activation(Activation.IDENTITY)
            .stride(1,1)
            .build())
        .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
            .kernelSize(2,2)
            .stride(2,2)
            .build())
        .layer(4, new DenseLayer.Builder()
            .activation(Activation.RELU)
            .nOut(500)
            .build())
        .layer(5, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(nOutput)
            .activation(Activation.SOFTMAX)
            .build())
        .setInputType(InputType.convolutionalFlat(28, 28, 1))
        .backprop(true).pretrain(false).build();
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    
    //train
    log.info("Training...");
    for (int i = 0; i < nEpochs; i++) {
      model.fit(trainData);
      log.info("Completed epoch {}", i);
    }
    
    //test
    log.info("Evaluating...");
    Evaluation eval = new Evaluation(nOutput);
    while (testData.hasNext()) {
      DataSet ds = testData.next();
      eval.eval(ds.getLabels(), model.output(ds.getFeatures()));
    }
    log.info(eval.stats());

  }

}
