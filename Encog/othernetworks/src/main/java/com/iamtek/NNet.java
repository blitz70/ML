package com.iamtek;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.art.ART1;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.pattern.ART1Pattern;
import org.encog.neural.pattern.ElmanPattern;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.neural.pattern.JordanPattern;
import org.encog.neural.pattern.NeuralNetworkPattern;

public class NNet {
	
	public static BasicNetwork createFFNet(){
		NeuralNetworkPattern pattern = new FeedForwardPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setInputNeurons(1);
		pattern.addHiddenLayer(6);
		pattern.setOutputNeurons(1);
		BasicNetwork network = (BasicNetwork)pattern.generate();
		network.reset();
		return network;
	}
	
	public static BasicNetwork createElmanNet(){
		NeuralNetworkPattern pattern = new ElmanPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setInputNeurons(1);
		pattern.addHiddenLayer(6);
		pattern.setOutputNeurons(1);
		BasicNetwork network = (BasicNetwork)pattern.generate();
		network.reset();
		return network;
	}
	
	public static BasicNetwork createJordanNet(){
		NeuralNetworkPattern pattern = new JordanPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setInputNeurons(1);
		pattern.addHiddenLayer(6);
		pattern.setOutputNeurons(1);
		BasicNetwork network = (BasicNetwork)pattern.generate();
		network.reset();
		return network;
	}

	public static ART1 createART1Net(int input, int output){
		NeuralNetworkPattern pattern = new ART1Pattern();
		pattern.setInputNeurons(input);
		pattern.setOutputNeurons(output);
		ART1 network = (ART1)pattern.generate();
		network.reset();
		return network;
	}

}
