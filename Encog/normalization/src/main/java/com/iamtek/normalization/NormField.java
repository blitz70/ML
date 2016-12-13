package com.iamtek.normalization;

import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

public class NormField {

	public static void main(String[] args) {

		NormalizedField fuelStats = new NormalizedField(NormalizationAction.Normalize, "fuel", 200, 0, 0.9, -0.9);
		
		double num = 100;
		double n = fuelStats.normalize(num);
		double d = fuelStats.deNormalize(n);
		
		System.out.println(num + " "+ n + " "+ d);
		
	}

}
