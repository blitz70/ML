package com.iamtek;

import java.io.File;
import java.util.Arrays;
import org.encog.ConsoleStatusReportable;
import org.encog.Encog;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.arrayutil.VectorWindow;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

public class PredictMain2 {
	
	public static String FILE_PATH = "src/main/resources/";
	public static String DATA_FILE = "data.txt";
	static final int WINDOW_LEAD = 1;
	static final int WINDOW_LAG = 10;

	public static void main(String[] args) {

		//map input
		CSVFormat csvFormat = new CSVFormat('.', ',');
		VersatileDataSource dataSource = new CSVDataSource(new File(FILE_PATH, DATA_FILE), true, csvFormat);
		VersatileMLDataSet dataSet = new VersatileMLDataSet(dataSource);
		NormalizationHelper helper = dataSet.getNormHelper();
		helper.setFormat(csvFormat);
		ColumnDefinition columnAmount = dataSet.defineSourceColumn("Amount", ColumnType.continuous);
		ColumnDefinition columnRate = dataSet.defineSourceColumn("Rate", ColumnType.continuous);
		ColumnDefinition columnOutput = dataSet.defineSourceColumn("Percent", ColumnType.nominal);
		dataSet.analyze();
		
		//setup model, normalization
		dataSet.defineInput(columnAmount);
		dataSet.defineInput(columnRate);
		dataSet.defineInput(columnOutput);
		dataSet.defineOutput(columnOutput);
		//dataSet.defineSingleOutputOthersInput(columnOutput);
		EncogModel model = new EncogModel(dataSet);
		model.selectMethod(dataSet, MLMethodFactory.TYPE_FEEDFORWARD);
		model.setReport(new ConsoleStatusReportable());
		dataSet.normalize();
		
		//fit model
		dataSet.setLeadWindowSize(WINDOW_LEAD);
		dataSet.setLagWindowSize(WINDOW_LAG);
		model.holdBackValidation(0.1, false, 1001);
		model.selectTrainingType(dataSet);
		MLRegression bestMethod = (MLRegression) model.crossvalidate(5, false);
		System.out.println(
				"Training error:" + model.calculateError(bestMethod, model.getTrainingDataset()) +
				", Validation error:" + model.calculateError(bestMethod, model.getValidationDataset())
		);
		System.out.println(helper.toString());
		System.out.println(
				"Final model:" + bestMethod + 
				", Inputs:" + bestMethod.getInputCount() + 
				", Outputs:"+bestMethod.getOutputCount());

		//predict
		ReadCSV csv = new ReadCSV(new File(FILE_PATH, DATA_FILE), true, csvFormat);
		String[] line = new String[3];
		double[] slice = new double[5];//2
		VectorWindow window = new VectorWindow(WINDOW_LEAD + WINDOW_LAG);
		MLData input = helper.allocateInputVector(WINDOW_LEAD + WINDOW_LAG);
		int count = 0;
		double oldValue = 0;
		int countDirectionHit = 0;
		int countDirectionMiss = 0;
		while(csv.next()){
			line[0] = csv.get(1);
			line[1] = csv.get(2);
			line[2] = csv.get(3);
			helper.normalizeInputVector(line, slice, false);
			if (window.isReady()) {
				window.copyWindow(input.getData(), 0);
				String correct = csv.get(3);
				MLData output = bestMethod.compute(input);
				String predicted = helper.denormalizeOutputVectorToString(output)[0];
				StringBuilder result = new StringBuilder();
				result.append(csv.get(0) + " " + oldValue + " " + Arrays.toString(line));
				result.append(" -> predicted:");
				result.append(predicted);
				result.append(" (correct:");
				result.append(correct);
				result.append(")");
				count++;
				//direction
				if(predicted.equals(correct)){
					countDirectionHit++;
					result.append(" Hit");
				} else {
					countDirectionMiss++;
					result.append(" Miss");
				}
				System.out.println(result.toString());
			}
			window.add(slice);
			oldValue = Double.parseDouble(line[2]);
		}
		System.out.println("Count:" + count + ", Hit:" + countDirectionHit + ", Miss:" + countDirectionMiss + " Hit%:" + countDirectionHit*100/(countDirectionHit+countDirectionMiss));	//50% hit
		//last prediction
		window.copyWindow(input.getData(), 0);
		MLData output = bestMethod.compute(input);
		String predicted = helper.denormalizeOutputVectorToString(output)[0];
		System.out.println("Next prediction: " + predicted);
		System.out.println(
				"Final model:" + bestMethod + 
				", Inputs:" + bestMethod.getInputCount() + 
				", Outputs:"+bestMethod.getOutputCount());
		EncogDirectoryPersistence.saveObject(new File(FILE_PATH, "nn2.txt"), bestMethod);
		Encog.getInstance().shutdown();
	}
	
}
