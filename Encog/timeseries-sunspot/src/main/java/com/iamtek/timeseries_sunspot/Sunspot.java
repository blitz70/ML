package com.iamtek.timeseries_sunspot;

import java.io.File;
import java.net.URISyntaxException;
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
import org.encog.util.arrayutil.VectorWindow;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

public class Sunspot {

	static final int WINDOW_LEAD = 1;	//future1
	static final int WINDOW_LAG = 3;	//past3 + today1
	
	public static void main(String[] args) {

			//String URL = "http://solarscience.msfc.nasa.gov/greenwch/spot_num.txt"
			long startTime = System.currentTimeMillis();

			//get input file
			File dataFile = null;
			try {
				dataFile = new File(Sunspot.class.getResource("/spot_num.txt").toURI());
			} catch (URISyntaxException e) {
				e.printStackTrace();
			}
			
			//map input
			CSVFormat csvFormat = new CSVFormat('.', ' ');
			VersatileDataSource dataSource = new CSVDataSource(dataFile, true, csvFormat);
			VersatileMLDataSet dataSet = new VersatileMLDataSet(dataSource);
			NormalizationHelper helper = dataSet.getNormHelper();
			helper.setFormat(csvFormat);
			ColumnDefinition columnSSN = dataSet.defineSourceColumn("SSN", 2, ColumnType.continuous);
			ColumnDefinition columnDEV = dataSet.defineSourceColumn("DEV", 3, ColumnType.continuous);
			/*helper.defineUnknownValue("?");
			helper.defineMissingHandler(columnHorsepower, new MeanMissingHandler());*/
			dataSet.analyze();
			
			//setup model, normalization
			//dataSet.defineSingleOutputOthersInput(columnSSN);
			dataSet.defineInput(columnSSN);
			dataSet.defineInput(columnDEV);
			dataSet.defineOutput(columnSSN);
			EncogModel model = new EncogModel(dataSet);
			model.selectMethod(dataSet, MLMethodFactory.TYPE_FEEDFORWARD);
			model.setReport(new ConsoleStatusReportable());
			dataSet.normalize();
			
			//fit model
			dataSet.setLeadWindowSize(WINDOW_LEAD);
			dataSet.setLagWindowSize(WINDOW_LAG);
			model.holdBackValidation(0.3, false, 1001);
			model.selectTrainingType(dataSet);
			MLRegression bestMethod = (MLRegression) model.crossvalidate(5, false);
			System.out.println(
					"Training error:" + model.calculateError(bestMethod, model.getTrainingDataset()) +
					", Validation error:" + model.calculateError(bestMethod, model.getValidationDataset())
			);
			System.out.println(helper.toString());
			System.out.println("Final model:" + bestMethod);
			
			//user model
			ReadCSV csv = new ReadCSV(dataFile, true, csvFormat);
			String[] line = new String[2];
			double[] slice = new double[2];
			VectorWindow window = new VectorWindow(WINDOW_LEAD + WINDOW_LAG);
			MLData input = helper.allocateInputVector(WINDOW_LEAD + WINDOW_LAG);
			double count = 0;
			double errorSum = 0;
			while(csv.next()){
				line[0] = csv.get(2);	//SSN
				line[1] = csv.get(3);	//DEV
				helper.normalizeInputVector(line, slice, false); //add record to slice
				if (window.isReady()) {
					//input <- window(accumulated past records), input does'nt have current record info
					window.copyWindow(input.getData(), 0);
					String correct = csv.get(2);
					MLData output = bestMethod.compute(input);
					String predicted = helper.denormalizeOutputVectorToString(output)[0];
					StringBuilder result = new StringBuilder();
					result.append(Arrays.toString(line));
					result.append(" -> predicted:");
					result.append(predicted);
					result.append(" (correct:");
					result.append(correct);
					result.append(")");
					/*Double error;
					if (correct.equals("0.0")){
						error = 1.0;
					} else {
						error = (Double.parseDouble(correct) - Double.parseDouble(predicted))/(Double.parseDouble(correct));
					}
					if (Math.abs(error) >= 1.0){
						error = 1.0;
					}*/
					Double error = (Double.parseDouble(correct) - Double.parseDouble(predicted))/(Double.parseDouble(predicted));
					result.append(" error:" + error);
					errorSum += Math.abs(error);
					count++;
					System.out.println(result.toString());
				}
				window.add(slice);	//add slice to window
				if (count >= 100) break;
			}
			System.out.println("Count:" + count + ", ErrorSum:" + errorSum*100 + "%, ErrorAve:" + errorSum/count*100 +"%"); //45%
			System.out.println(System.currentTimeMillis()-startTime);
			Encog.getInstance().shutdown();
			
	}

}
