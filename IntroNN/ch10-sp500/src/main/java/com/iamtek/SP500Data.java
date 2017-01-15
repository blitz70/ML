package com.iamtek;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.util.Date;
import java.util.Set;
import java.util.TreeSet;

import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

public class SP500Data {
	
	//http://finance.yahoo.com/quote/%5EGSPC/history?p=^GSPC
	//https://fred.stlouisfed.org/categories/117/downloaddata
	
	private Set<FinancialSample> samples = new TreeSet<FinancialSample>();
	private Set<InterestRate> rates = new TreeSet<InterestRate>();
	private int inputSize;
	private int outputSize;

	public SP500Data(int inputSize, int outputSize) {
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
				//double percent = Math.signum(sample.getAmount() - prev);//updown
				sample.setPercent(percent);
			}
			prev = sample.getAmount();
		}
	}
	
	public void getInputData(int offset, double[] input){
		Object[] samplesArray = this.samples.toArray();
		for (int i = 0; i < this.inputSize; i++) {
			FinancialSample sample = (FinancialSample) samplesArray[offset+i];
			input[2*i] = sample.getPercent();
			input[2*i+1] = sample.getRate();
		}
		/*for (int i = 0; i < this.inputSize; i++) {
			FinancialSample sample = (FinancialSample) samplesArray[offset+i];
			input[i] = sample.getPercent();
			input[i+this.outputSize] = sample.getRate();	//?
		}*/
	}
	
	public void getOutputData(int offset, double[] output){
		Object[] samplesArray = this.samples.toArray();
		for (int i = 0; i < this.outputSize; i++) {
			FinancialSample sample = (FinancialSample) samplesArray[offset+i+this.inputSize];//?
			output[i] = sample.getPercent();
		}
	}
	
	public double getCurrentRate(Date date){
		//search dates till dates passes required one, return value of date before the passed date
		double currentRate = 0;
		for (InterestRate rate : this.rates) {
			if(rate.getDate().after(date)){
				return currentRate;
			} else {
				currentRate = rate.getRate();
			}
		}
		return currentRate;
	}
	
	public void stitchRates(){
		for (FinancialSample sample : this.samples) {
			double rate = getCurrentRate(sample.getDate());
			sample.setRate(rate);
		}
	}
	
	public int size(){
		return this.samples.size();
	}
	
	public void generateData(String filePath, String financeFile, String rateFile) throws IOException, ParseException{
		ReadCSV csv = new ReadCSV(filePath+financeFile, true, CSVFormat.DECIMAL_POINT);
		while(csv.next()){
			Date date = csv.getDate("date");
			double amount = csv.getDouble("adj close");
			FinancialSample sample = new FinancialSample();
			sample.setDate(date);
			sample.setAmount(amount);
			this.samples.add(sample);
		}
		csv = new ReadCSV(filePath+rateFile, true, CSVFormat.DECIMAL_POINT);
		while(csv.next()){
			Date date = csv.getDate("date");
			double rate = csv.getDouble("value");
			InterestRate ir = new InterestRate(date, rate);
			this.rates.add(ir);
		}
		csv.close();
		stitchRates();
		calculatePercents();
	}
	
	public void saveData(String filePath, String fileName) throws IOException{
		OutputStream os = new FileOutputStream(new File(filePath,fileName));
		DataOutputStream dos = new DataOutputStream(os);
		DecimalFormat fm = new DecimalFormat("##.####");
		StringBuilder result = new StringBuilder();
		result.append("Date,Amount,Rate,Percent");
		dos.writeBytes(result.toString());
		for (FinancialSample sample : samples) {
			result = new StringBuilder();
			result.append("\n"+ReadCSV.displayDate(sample.getDate()));
			result.append(",");
			result.append(fm.format(sample.getAmount()));
			result.append(",");
			result.append(sample.getRate());
			result.append(",");
			result.append(fm.format(sample.getPercent()));
			dos.writeBytes(result.toString());
		}
		dos.close();
		os.close();
	}
	
}
