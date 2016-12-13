package com.iamtek.normalization;

import java.io.File;

import org.encog.Encog;
import org.encog.app.analyst.AnalystFileFormat;
import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.csv.normalize.AnalystNormalizeCSV;
import org.encog.app.analyst.script.normalize.AnalystField;
import org.encog.app.analyst.wizard.AnalystWizard;
import org.encog.util.csv.CSVFormat;

public class NormFile {

	public static void main(String[] args) {

		File sourceFile = null, targetFile = null;
		try {
			sourceFile = new File("src/main/resources/source.data");
			targetFile = new File("src/main/resources/target.data");
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		EncogAnalyst analyst = new EncogAnalyst();
		AnalystWizard wizard = new AnalystWizard(analyst);
		wizard.wizard(sourceFile, true, AnalystFileFormat.DECPNT_COMMA);
		dumpFieldInfo(analyst);
		
		AnalystNormalizeCSV norm = new AnalystNormalizeCSV();
		norm.analyze(sourceFile, true, CSVFormat.ENGLISH, analyst);
		norm.setProduceOutputHeaders(true);
		norm.normalize(targetFile);
		
		try {
			analyst.save(new File("src/main/resources/stats.ega"));
		} catch (Exception e) {
			e.printStackTrace();
		}
		analyst.load(new File("src/main/resources/stats.ega"));
		
		Encog.getInstance().shutdown();
		
	}
	
	public static void dumpFieldInfo(EncogAnalyst analyst){
		System.out.println("Fields found in file:");
		for (AnalystField field : analyst.getScript().getNormalize().getNormalizedFields()) {
			StringBuilder line = new StringBuilder();
			line.append(field.getName());
			line.append(",action=");
			line.append(field.getAction());
			line.append(",min=");
			line.append(field.getActualLow());
			line.append(",max=");
			line.append(field.getActualHigh());
			System.out.println(line.toString());
		}
	}

}
