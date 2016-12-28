package com.iamtek;

import java.awt.Image;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.encog.EncogError;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.train.strategy.ResetStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.platformspecific.j2se.data.image.ImageMLData;
import org.encog.platformspecific.j2se.data.image.ImageMLDataSet;
import org.encog.util.downsample.Downsample;
import org.encog.util.downsample.RGBDownsample;
import org.encog.util.downsample.SimpleIntensityDownsample;
import org.encog.util.simple.EncogUtility;

class ImagePair {

	private File file;
	private int identity;
	
	public ImagePair(File file, int identity) {
		this.file = file;
		this.identity = identity;
	}

	public File getFile() {
		return file;
	}

	public int getIdentity() {
		return identity;
	}

}

public class ImageMain {

	private static final String IMAGE_PATH = "";
	
	//?
	private Map<String, String> args = new HashMap<String, String>();
	private String line;
	private Map<String, Integer> identity2neuron = new HashMap<String, Integer>();
	private Map<Integer, String> neuron2identity = new HashMap<Integer, String>();
	
	//image
	private int downsampleWidth;
	private int downsampleHeight;
	private Downsample downsample;
	private MLDataSet training;
	private List<ImagePair> imageList = new ArrayList<ImagePair>();
	
	//network
	private int outputCount;
	private BasicNetwork network;

	public static void main(String[] args) {
	}

	private void processCreateTraining(){
		String width = getArg("width");
		String height = getArg("height");
		String type = getArg("type");
		this.downsampleWidth = Integer.parseInt(width);
		this.downsampleHeight = Integer.parseInt(height);
		if (type.equals("RGB")) {
			this.downsample = new RGBDownsample();
		} else {
			this.downsample = new SimpleIntensityDownsample();
		}
		this.training = new ImageMLDataSet(downsample, false, 1, -1);
		System.out.println("Training set created");
	}
	
	private void processInput(){
		String image = getArg("image");
		String identity = getArg("identity");
		int index = assignIdentity(identity);
		File file = new File(IMAGE_PATH, image);
		this.imageList.add(new ImagePair(file, index));
		System.out.println("Added input image:" + image);
	}

	private void processNetwork() throws IOException{
		System.out.println("Downsampling images...");
		for (ImagePair pair : this.imageList) {
			MLData ideal = new BasicMLData(this.outputCount);
			int index = pair.getIdentity();
			for (int i = 0; i < this.outputCount; i++) {
				if(i == index){
					ideal.setData(i, 1);
				} else {
					ideal.setData(i, -1);
				}
			}
			Image image = ImageIO.read(pair.getFile());
			MLData data = new ImageMLData(image);
			this.training.add(data, ideal);
		}
		int hidden1 = Integer.parseInt(getArg("hidden1"));
		int hidden2 = Integer.parseInt(getArg("hidden2"));
		((ImageMLDataSet) this.training).downsample(this.downsampleHeight, this.downsampleWidth);
		this.network = EncogUtility.simpleFeedForward(
				this.training.getInputSize(),
				hidden1, hidden2,
				this.training.getIdealSize(), true);
		System.out.println("Created network: " + this.network.toString());
	}
	
	private void processTrain(){
		String mode = getArg("mode");
		int minutes = Integer.parseInt(getArg("minutes"));
		double strategyError = Double.parseDouble(getArg("strategyerror"));
		int strategyCycles = Integer.parseInt(getArg("strategycycles"));
		System.out.println("Training begining... Output patterns=" + this.outputCount);
		ResilientPropagation train = new ResilientPropagation(this.network, this.training);
		train.addStrategy(new ResetStrategy(strategyError, strategyCycles));
		EncogUtility.trainConsole(train, this.network, this.training, minutes);
		System.out.println("Training stopped...");
	}
	
	private int assignIdentity(String identity) {
		if(this.identity2neuron.containsKey(identity.toLowerCase())){
			return this.identity2neuron.get(identity.toLowerCase());
		}
		int result = this.outputCount;
		this.identity2neuron.put(identity.toLowerCase(), result);
		this.neuron2identity.put(result, identity.toLowerCase());
		this.outputCount++;
		return result;
	}

	private String getArg(String name) {
		String result = this.args.get(name);
		if(result == null){
			throw new EncogError("Missing argument " + name + " on line" + this.line);
		}
		return result;
	}

}
