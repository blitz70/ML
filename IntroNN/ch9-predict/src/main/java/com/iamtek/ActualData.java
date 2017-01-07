package com.iamtek;

public class ActualData {

	private double[] actual;
	private int inputSize;
	private int outputSize;
	
	public ActualData(int size, int inputSize, int outputSize) {
		this.actual = new double[size];
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		int angle = 0;
		for (int i = 0; i < this.actual.length; i++) {
			this.actual[i] = sineFunc(angle);
			angle += 10;
		}
	}

	private double sineFunc(int angle) {
		return Math.sin(Math.toRadians(angle));
	}
	
	public double[] getInputData(int offset){
		double[] result = new double[this.inputSize];
		for (int i = 0; i < this.inputSize; i++) {
			result[i] = this.actual[offset+i];
		}
		return result;
	}

	public double[] getOutputData(int offset){
		double[] result = new double[this.outputSize];
		for (int i = 0; i < this.outputSize; i++) {
			result[i] = this.actual[offset+this.inputSize+i];
		}
		return result;
	}
}
