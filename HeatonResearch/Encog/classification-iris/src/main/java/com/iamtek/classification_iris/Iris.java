package com.iamtek.classification_iris;

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
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;

public class Iris {

	public static void main(String[] args) {

		//input
		File file = null;
		try {
			file = new File(Iris.class.getResource("/iris.data").toURI());
		} catch (URISyntaxException e) {
			e.printStackTrace();
		}
		VersatileDataSource source = new CSVDataSource(file, false, CSVFormat.DECIMAL_POINT);
		VersatileMLDataSet data = new VersatileMLDataSet(source);
		data.defineSourceColumn("sepal-length", 0, ColumnType.continuous);
		data.defineSourceColumn("sepal-width", 1, ColumnType.continuous);
		data.defineSourceColumn("petal-length", 2, ColumnType.continuous);
		data.defineSourceColumn("petal-width", 3, ColumnType.continuous);
		ColumnDefinition outputColumn = data.defineSourceColumn("species", 4, ColumnType.nominal);
		data.analyze();
		
		//decide model, normalize data, prepare to validate
		data.defineSingleOutputOthersInput(outputColumn);
		EncogModel model = new EncogModel(data);
		model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);
		model.setReport(new ConsoleStatusReportable());
		data.normalize();
		model.holdBackValidation(0.3, true, 1001);
		model.selectTrainingType(data);
		MLRegression bestMethod = (MLRegression) model.crossvalidate(5, true);
		
		//display
		System.out.println("Training error: " + EncogUtility.calculateRegressionError(bestMethod, model.getTrainingDataset()));
		System.out.println("Validate error: " + EncogUtility.calculateRegressionError(bestMethod, model.getValidationDataset()));
		NormalizationHelper helper = data.getNormHelper();
		System.out.println(helper.toString());
		System.out.println("Final model: " + bestMethod);
		
		//save
		ReadCSV csv = new ReadCSV(file, false, CSVFormat.DECIMAL_POINT);
		String [] line = new String[4];
		MLData input = helper.allocateInputVector();
		while(csv.next()) {
			StringBuilder result = new StringBuilder();
			line[0] = csv.get(0);
			line[1] = csv.get(1);
			line[2] = csv.get(2);
			line[3] = csv.get(3);
			String correct = csv.get(4);
			helper.normalizeInputVector(line, input.getData(), false);
			MLData output = bestMethod.compute(input);
			String irisChosen = helper.denormalizeOutputVectorToString(output)[0];
			result.append(Arrays.toString(line));
			result.append(" -> predicted: ");
			result.append(irisChosen);
			result.append("(correct: ");
			result.append(correct);
			result.append(")");
			System.out.println(result.toString());
		}
		Encog.getInstance().shutdown();
		
	}

}
