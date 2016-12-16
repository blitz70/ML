package com.iamtek.train_supervised;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.randomize.NguyenWidrowRandomizer;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.genetic.NeuralGeneticAlgorithm;

public class LanderMain {

	public static BasicNetwork createNet(){
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, 3));
		network.addLayer(new BasicLayer(new ActivationTANH(), true, 50));
		network.addLayer(new BasicLayer(1));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}
	
	public static void main(String[] args) {
		BasicNetwork network = createNet();
		/*MLTrain train = new MLMethodGeneticAlgorithm(new MethodFactory() {
			@Override
			public MLMethod factor() {
				return createNet();
			}
		}, new PilotScore(), 500);*/
		MLTrain train = new NeuralGeneticAlgorithm(network, new NguyenWidrowRandomizer(), new PilotScore(), 500, 0.1, 0.25);
		for (int i = 0; i < 100; i++) {
			train.iteration();
			System.out.println("Epoch #" + i + ", Error:" + train.getError());
		}
		train.finishTraining();
		System.out.println("\nHow the winning network landed:");
		network = (BasicNetwork) train.getMethod();
		NeuralPilot pilot = new NeuralPilot(network, true);
		System.out.println(pilot.scorePilot());
		Encog.getInstance().shutdown();
	}

}
