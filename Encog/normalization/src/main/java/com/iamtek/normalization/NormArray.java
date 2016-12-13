package com.iamtek.normalization;

import java.util.Arrays;

import org.encog.util.arrayutil.NormalizeArray;

public class NormArray {

	public static void main(String[] args) {

		NormalizeArray norm = new NormalizeArray();
		
		norm.setNormalizedHigh(1);
		norm.setNormalizedLow(-1);
		double[] rawDataArray = {32, 22};
		double[] normalizedSunspots = norm.process(rawDataArray);
		
		System.out.println(Arrays.toString(rawDataArray) + " " + Arrays.toString(normalizedSunspots));
	}

}
