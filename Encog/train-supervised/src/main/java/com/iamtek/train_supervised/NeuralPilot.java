package com.iamtek.train_supervised;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

public class NeuralPilot {

	private BasicNetwork network;
	private boolean track;
	private boolean thrust;
	private NormalizedField fuel;
	private NormalizedField altitude;
	private NormalizedField velocity;

	public NeuralPilot(BasicNetwork network, boolean track) {
		this.network = network;
		this.track = track;
		this.fuel = new NormalizedField(NormalizationAction.Normalize, "fuel", 200, 0, 0.9, -0.9);
		this.altitude = new NormalizedField(NormalizationAction.Normalize, "altitude", 10000, 0, 0.9, -0.9);
		this.velocity = new NormalizedField(NormalizationAction.Normalize, "velocity", LanderSimulator.TERMINAL_VELOCITY, -LanderSimulator.TERMINAL_VELOCITY, 0.9, -0.9);
	}

	public double scorePilot() {
		LanderSimulator sim = new LanderSimulator();
		while(sim.isFly()){
			MLData input = new BasicMLData(3);
			input.setData(0, fuel.normalize(sim.getFuel()));
			input.setData(1, altitude.normalize(sim.getAltitude()));
			input.setData(2, velocity.normalize(sim.getVelocity()));
			MLData output = network.compute(input);
			if(output.getData(0)>0){
				thrust = true;
				if(track) System.out.println("THRUST");
			} else thrust = false;
			sim.turn(thrust);
			if(track) System.out.println(sim.telemtry());
		}
		return sim.score();
	}

}
