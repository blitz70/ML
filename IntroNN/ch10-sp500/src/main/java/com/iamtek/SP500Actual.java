package com.iamtek;

import java.util.Date;
import java.util.Set;
import java.util.TreeSet;

public class SP500Actual {

	private Set<InterestRate> rates = new TreeSet<InterestRate>();
	private Set<FinancialSample> samples = new TreeSet<FinancialSample>();
	private int inputSize;
	private int outputSize;

	public SP500Actual(int inputSize, int outputSize) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
	}
	
	public Set<FinancialSample> getSamples() {
		return this.samples;
	}

	public void calculatePercents(){
		double prev = -1;
		for (FinancialSample sample : this.samples) {
			if(prev != -1){
				double percent = (sample.getAmount() - prev)/prev;
				sample.setPercent(percent);
			}
			prev = sample.getAmount();
		}
	}
	
	public void getInputData(int offset, double[] input){
		Object[] samplesArray = this.samples.toArray();
		for (int i = 0; i < this.inputSize; i++) {
			FinancialSample sample = (FinancialSample) samplesArray[offset+i];
			input[i] = sample.getPercent();
			input[i+this.outputSize] = sample.getRate();	//?
		}
	}
	
	public void getOutputData(int offset, double[] output){
		Object[] samplesArray = this.samples.toArray();
		for (int i = 0; i < this.outputSize; i++) {
			FinancialSample sample = (FinancialSample) samplesArray[offset+i+this.inputSize];//?
			output[i] = sample.getPercent();
		}
	}
	
	public double getPrimeRate(Date date){
		double currentRate = 0;
		for (InterestRate rate : this.rates) {
			if(rate.getEffectiveDate().after(date)){
				return currentRate;
			} else{
				currentRate = rate.getRate();
			}
		}
		return currentRate;
	}
	
	
	
}
