package com.iamtek.regression_mpg;

import java.io.File;
import java.net.URISyntaxException;
import java.util.Arrays;

import org.encog.ConsoleStatusReportable;
import org.encog.ensemble.training.BackpropagationFactory;
import org.encog.ml.MLRegression;
import org.encog.ml.TrainingImplementationType;
import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.factory.MLTrainFactory;
import org.encog.ml.model.EncogModel;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

public class Mpg {

	public static void main(String[] args) {

		//String URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
		long startTime = System.currentTimeMillis();

		//get input file
		File dataFile = null;
		try {
			dataFile = new File(Mpg.class.getResource("/auto-mpg.data").toURI());
		} catch (URISyntaxException e) {
			e.printStackTrace();
		}
		//map input
		CSVFormat csvFormat = new CSVFormat('.', ' ');
		VersatileDataSource dataSource = new CSVDataSource(dataFile, false, csvFormat);
		VersatileMLDataSet dataSet = new VersatileMLDataSet(dataSource);
		NormalizationHelper helper = dataSet.getNormHelper();
		helper.setFormat(csvFormat);
		/*
		mpg				continuous
		cylinders		multi-valued discrete
		displacement	continuous
		horsepower		continuous
		weight			continuous
		acceleration	continuous
		model-year		multi-valued discrete
		origin			multi-valued discrete
		car-name		string, no def needed
		 */
		ColumnDefinition columnMPG = dataSet.defineSourceColumn("MPG", 0, ColumnType.continuous);
		ColumnDefinition columnCylinders = dataSet.defineSourceColumn("Cylinders", 1, ColumnType.ordinal);
		columnCylinders.defineClass(new String[] {"3", "4", "5", "6", "8"});
		ColumnDefinition columnDisplacement = dataSet.defineSourceColumn("Displacement", 2, ColumnType.continuous);
		ColumnDefinition columnHorsepower = dataSet.defineSourceColumn("Horsepower", 3, ColumnType.continuous);
		ColumnDefinition columnWeight = dataSet.defineSourceColumn("Weight", 4, ColumnType.continuous);
		ColumnDefinition columnAcceleration = dataSet.defineSourceColumn("Acceleration", 5, ColumnType.continuous);
		ColumnDefinition columnModelYear = dataSet.defineSourceColumn("ModelYear", 6, ColumnType.ordinal);
		columnModelYear.defineClass(new String[] {"70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82"});
		ColumnDefinition columnOrigin = dataSet.defineSourceColumn("Origin", 7, ColumnType.nominal);
		/*helper.defineUnknownValue("?");
		helper.defineMissingHandler(columnHorsepower, new MeanMissingHandler());*/
		dataSet.analyze();
		System.out.println("Analyze");
		System.out.println(System.currentTimeMillis()-startTime);
		
		//model, normalize
		dataSet.defineSingleOutputOthersInput(columnMPG);
		EncogModel model = new EncogModel(dataSet);
		model.selectMethod(dataSet, MLMethodFactory.TYPE_FEEDFORWARD);
		model.setReport(new ConsoleStatusReportable());
		dataSet.normalize();
		System.out.println("Model");
		System.out.println(System.currentTimeMillis()-startTime);
		
		//fit model
		model.holdBackValidation(0.3, true, 1001);
		model.selectTrainingType(dataSet);
		MLRegression bestMethod = (MLRegression) model.crossvalidate(5, true);
		System.out.println(
				"Training error:" + model.calculateError(bestMethod, model.getTrainingDataset()) +
				", Validation error:" + model.calculateError(bestMethod, model.getValidationDataset())
		);
		System.out.println(helper.toString());
		System.out.println("Final model:" + bestMethod);
		System.out.println("Fit");
		System.out.println(System.currentTimeMillis()-startTime);
		
		//use model
		ReadCSV csv = new ReadCSV(dataFile, false, csvFormat);
		MLData input = helper.allocateInputVector();
		String[] line = new String[7];
		double predictedSum = 0;
		double correctSum = 0;
		double count = 0;
		while(csv.next()){
			String correct = csv.get(0);
			line[0] = csv.get(1);
			line[1] = csv.get(2);
			line[2] = csv.get(3);
			line[3] = csv.get(4);
			line[4] = csv.get(5);
			line[5] = csv.get(6);
			line[6] = csv.get(7);
			helper.normalizeInputVector(line, input.getData(), false);
			MLData output = bestMethod.compute(input);
			String predictedMpg = helper.denormalizeOutputVectorToString(output)[0];
			StringBuilder result = new StringBuilder();
			result.append(Arrays.toString(line));
			result.append("\t-> predicted:");
			result.append(predictedMpg);
			result.append(" (correct:");
			result.append(correct);
			result.append(")");
			System.out.println(result.toString());
			predictedSum += Double.parseDouble(predictedMpg);
			correctSum += Double.parseDouble(correct);
			count ++;
		}
		System.out.println(correctSum + ", " + predictedSum + ", " + count);
		System.out.println((correctSum-predictedSum)/count*100+"%");

	}

}
