package com.iamtek.timeseries_stock;

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

public class Stock {

	static final int WINDOW_LEAD = 1;
	static final int WINDOW_LAG = 3;
	
	public static void main(String[] args) {

			//get input file
			File dataFile = null;
			try {
				dataFile = new File(Stock.class.getResource("/kospi.data").toURI());
			} catch (URISyntaxException e) {
				e.printStackTrace();
			}
			
			//map input
			CSVFormat csvFormat = new CSVFormat('.', ',');
			VersatileDataSource dataSource = new CSVDataSource(dataFile, true, csvFormat);
			VersatileMLDataSet dataSet = new VersatileMLDataSet(dataSource);
			NormalizationHelper helper = dataSet.getNormHelper();
			helper.setFormat(csvFormat);
			ColumnDefinition columnOpen = dataSet.defineSourceColumn("Open", ColumnType.continuous);
			ColumnDefinition columnHigh = dataSet.defineSourceColumn("High", ColumnType.continuous);
			ColumnDefinition columnLow = dataSet.defineSourceColumn("Low", ColumnType.continuous);
			ColumnDefinition columnClose = dataSet.defineSourceColumn("Close", ColumnType.continuous);
			ColumnDefinition columnVolume = dataSet.defineSourceColumn("Volume", ColumnType.continuous);
			ColumnDefinition columnAdjClose = dataSet.defineSourceColumn("Adj Close", ColumnType.continuous);
			dataSet.analyze();
			
			//setup model, normalization
			//dataSet.defineSingleOutputOthersInput(columnSSN);
			dataSet.defineInput(columnOpen);
			dataSet.defineInput(columnHigh);
			dataSet.defineInput(columnLow);
			dataSet.defineInput(columnClose);
			dataSet.defineInput(columnVolume);
			dataSet.defineInput(columnAdjClose);
			dataSet.defineOutput(columnAdjClose);
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
			String[] line = new String[6];
			double[] slice = new double[6];
			/*String[] line = new String[1];
			double[] slice = new double[1];*/
			VectorWindow window = new VectorWindow(WINDOW_LEAD + WINDOW_LAG);
			MLData input = helper.allocateInputVector(WINDOW_LEAD + WINDOW_LAG);
			double count = 0;
			double errorSum = 0;
			while(csv.next()){
				//if (count >= 100) break;
				line[0] = csv.get(1);
				line[1] = csv.get(2);
				line[2] = csv.get(3);
				line[3] = csv.get(4);
				line[4] = csv.get(5);
				line[5] = csv.get(6);
				/*line[0] = csv.get(6);*/
				helper.normalizeInputVector(line, slice, false);
				if (window.isReady()) {
					window.copyWindow(input.getData(), 0);
					String correct = csv.get(6);
					MLData output = bestMethod.compute(input);
					String predicted = helper.denormalizeOutputVectorToString(output)[0];
					StringBuilder result = new StringBuilder();
					result.append(csv.get(0) + " " + Arrays.toString(line));
					result.append(" -> predicted:");
					result.append(predicted);
					result.append(" (correct:");
					result.append(correct);
					result.append(")");
					Double error = (Double.parseDouble(correct) - Double.parseDouble(predicted))/Double.parseDouble(correct);
					result.append(" error:" + error);
					errorSum += Math.abs(error);
					count++;
					System.out.println(result.toString());
				}
				window.add(slice);
			}
			System.out.println("Count:" + count + ", ErrorSum:" + errorSum*100 + "%, ErrorAve:" + errorSum/count*100 +"%"); //5%
			Encog.getInstance().shutdown();
			
	}

}
