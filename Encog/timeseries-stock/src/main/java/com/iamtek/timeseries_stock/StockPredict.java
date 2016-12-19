package com.iamtek.timeseries_stock;

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
import org.encog.util.arrayutil.VectorWindow;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

public class StockPredict {

	static final int WINDOW_LEAD = 1;
	static final int WINDOW_LAG = 4;
	
	public void run(File dataFile) {

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
			model.holdBackValidation(0.1, false, 1001);
			model.selectTrainingType(dataSet);
			MLRegression bestMethod = (MLRegression) model.crossvalidate(5, false);
			System.out.println(
					"Training error:" + model.calculateError(bestMethod, model.getTrainingDataset()) +
					", Validation error:" + model.calculateError(bestMethod, model.getValidationDataset())
			);
			//System.out.println(helper.toString());
			System.out.println("Final model:" + bestMethod);
			
			//user model
			ReadCSV csv = new ReadCSV(dataFile, true, csvFormat);
			String[] line = new String[6];
			double[] slice = new double[6];
			/*String[] line = new String[1];
			double[] slice = new double[1];*/
			VectorWindow window = new VectorWindow(WINDOW_LEAD + WINDOW_LAG);
			MLData input = helper.allocateInputVector(WINDOW_LEAD + WINDOW_LAG);
			int count = 0;
			int countRecord = 0;
			double errorSum = 0;
			double oldValue = 0;
			int countDirectionHit = 0;
			int countDirectionMiss = 0;
			while(csv.next()){
				//if (count >= 100) break;
				countRecord ++;
				//if (countRecord < 4500) continue;
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
					result.append(csv.get(0) + " " + oldValue + " " + Arrays.toString(line));
					result.append(" -> predicted:");
					result.append(predicted);
					result.append(" (correct:");
					result.append(correct);
					result.append(")");
					double correctN = Double.parseDouble(correct);
					double predictedN = Double.parseDouble(predicted);
					Double error = (correctN - predictedN)/correctN;
					result.append(" error:" + error);
					errorSum += Math.abs(error);
					count++;
					//direction
					if(Math.signum(correctN-oldValue)*Math.signum(predictedN-oldValue) >= 0){
						countDirectionHit++;
						result.append(" Hit");
					} else {
						countDirectionMiss++;
						result.append(" Miss");
					}
					System.out.println(result.toString());
				}
				window.add(slice);
				oldValue = Double.parseDouble(line[5]);
			}
			System.out.println("Count:" + count + ", ErrorSum:" + errorSum*100 + "%, ErrorAve:" + errorSum*100/count +"%"); //4% error
			System.out.println("Hit:" + countDirectionHit + " Miss:" + countDirectionMiss + " Hit%:" + countDirectionHit*100/(countDirectionHit+countDirectionMiss));	//50% hit
			//last prediction
			window.copyWindow(input.getData(), 0);
			MLData output = bestMethod.compute(input);
			String predicted = helper.denormalizeOutputVectorToString(output)[0];
			System.out.println("Next prediction: " + predicted);
			Encog.getInstance().shutdown();
			
	}

}
