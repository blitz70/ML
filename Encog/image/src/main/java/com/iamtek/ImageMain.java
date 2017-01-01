package com.iamtek;

import java.awt.Image;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import javax.imageio.ImageIO;

import org.encog.Encog;
import org.encog.EncogError;
import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.HybridStrategy;
import org.encog.ml.train.strategy.ResetStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
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

	//?
	private Map<String, String> args = new HashMap<String, String>();
	private String line;
	private Map<String, Integer> identity2neuron = new HashMap<String, Integer>();
	private Map<Integer, String> neuron2identity = new HashMap<Integer, String>();
	
	//image
	/*private String trainingPath = "src/main/resources/training/";
	private String testPath = "src/main/resources/test/";*/
	private String trainingPath = "d:/code/digits/training/";
	private String testPath = "d:/code/digits/test/";
	private int downsampleWidth;
	private int downsampleHeight;
	private Downsample downsample;
	private MLDataSet training;
	private List<ImagePair> imageList = new ArrayList<ImagePair>();
	
	//network
	private int outputCount;
	private BasicNetwork network;

	public static void main(String[] args) {
		
		ImageMain prog = new ImageMain();
		try {
			prog.processScript();
		} catch (IOException e) {
			e.printStackTrace();
		}
		Encog.getInstance().shutdown();
	}
	
	private void processScript() throws IOException{
		File script = new File("src/main/resources/script2.txt");
		FileInputStream fstream = new FileInputStream(script);
		DataInputStream in = new DataInputStream(fstream);
		BufferedReader br = new BufferedReader(new InputStreamReader(in));
		while((this.line = br.readLine()) != null){
			int index  = this.line.indexOf(':');
			String command = this.line.substring(0, index).toLowerCase().trim();
			String args = this.line.substring(index+1).trim();
			StringTokenizer token = new StringTokenizer(args, ","); 
			this.args.clear();
			while(token.hasMoreTokens()){
				String arg = token.nextToken();
				int index2 = arg.indexOf(':');
				String key = arg.substring(0, index2).toLowerCase().trim();
				String value = arg.substring(index2+1).toLowerCase().trim();
				this.args.put(key, value);
			}
			if (command.equals("input")) processInput();
			if (command.equals("createtraining")) processCreateTraining();
			if (command.equals("train")) processTrain();
			if (command.equals("network")) processNetwork();
			if (command.equals("whatis")) processWhatIs();
		}
		br.close();
		in.close();
		fstream.close();
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
		/*String image = getArg("image");
		String identity = getArg("identity");
		int index = assignIdentity(identity);
		File file = new File(trainingPath, image);
		this.imageList.add(new ImagePair(file, index));
		System.out.println("Added input image:" + image);*/
		
		File input = null;
		for (int id = 0; id < 10; id++) {
			int no = 1;
			while(true){
				input = new File(trainingPath, "digits" + id + "-"+no+".jpg");
				if(!input.exists()) break;
				int index = assignIdentity(String.valueOf(id));
				this.imageList.add(new ImagePair(input, index));
				System.out.println("Added input image:" + input);
				no++;
			}
		}
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
		int minutes = Integer.parseInt(getArg("minutes"));
		double strategyError = Double.parseDouble(getArg("strategyerror"));
		int strategyCycles = Integer.parseInt(getArg("strategycycles"));
		System.out.println("Training begining... Output patterns=" + this.outputCount);
		MLTrain train = new ResilientPropagation(this.network, this.training);
		CalculateScore score = new TrainingSetScore(this.training);
		MLTrain sub = new NeuralSimulatedAnnealing(this.network, score, 10, 2, strategyCycles);
		train.addStrategy(new HybridStrategy(sub));
		train.addStrategy(new ResetStrategy(strategyError, strategyCycles));
		EncogUtility.trainConsole(train, this.network, this.training, minutes);
		System.out.println("Training stopped...");
	}
	
	private void processWhatIs() throws IOException{
		String filename = getArg("image");
		File file = new File(testPath, filename);
		Image img = ImageIO.read(file);
		MLData input = new ImageMLData(img);
		((ImageMLData)input).downsample(
				this.downsample, false,
				this.downsampleHeight, this.downsampleWidth,
				1, -1);
		int winner = this.network.winner(input);
		System.out.println("What is:"+filename+", looks like:"+this.neuron2identity.get(winner));
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
