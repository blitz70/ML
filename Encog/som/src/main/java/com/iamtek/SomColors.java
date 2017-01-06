package com.iamtek;

import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

import org.encog.mathutil.randomize.RangeRandomizer;
import org.encog.mathutil.rbf.RBFEnum;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.som.SOM;
import org.encog.neural.som.training.basic.BasicTrainSOM;
import org.encog.neural.som.training.basic.neighborhood.NeighborhoodRBF;

public class SomColors extends JFrame implements Runnable {

	private static final long serialVersionUID = 1L;

	private MapPanel map;
	private Thread thread;
	
	private SOM network;
	private BasicTrainSOM train;
	private NeighborhoodRBF rbf;
	private int iteration = 10000;
	private double startRate = 0.8;
	private double endRate = 0.03;
	private double startRadius = 10;
	private double endRadius = 1;

	public static void main(String[] args) {
		SomColors prog = new SomColors();
		prog.setVisible(true);

	}

	public SomColors() {
		this.setSize(640, 480);
		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
		this.network = createNetwork();
		this.getContentPane().add(map = new MapPanel(this));
		this.rbf = new NeighborhoodRBF(RBFEnum.Gaussian, MapPanel.COLS, MapPanel.ROWS);
		this.train = new BasicTrainSOM((SOM)this.network, 0.01, null, rbf);
		this.train.setForceWinner(false);
		this.thread = new Thread(this);
		thread.start();
	}

	private SOM createNetwork() {
		SOM result = new SOM(3, MapPanel.COLS*MapPanel.ROWS);
		result.reset();;
		return result;
	}

	public SOM getNetwork() {
		return this.network;
	}

	@Override
	public void run() {
		List<MLData> samples = new ArrayList<MLData>();
		for (int i = 0; i < 256; i++) {
			MLData data = new BasicMLData(3);
			data.setData(0, RangeRandomizer.randomize(-1, 1));
			data.setData(1, RangeRandomizer.randomize(-1, 1));
			data.setData(2, RangeRandomizer.randomize(-1, 1));
			samples.add(data);
		}
		this.train.setAutoDecay(iteration, startRate, endRate, startRadius, endRadius);
		for(int i = 0; i < iteration; i++){
			int index = (int) (Math.random()*samples.size());
			MLData pattern = samples.get(index);
			this.train.trainPattern(pattern);
			this.train.autoDecay();
			this.map.repaint();
			System.out.println("Iteration "+i+", "+this.train.toString());
		}
	}

}
