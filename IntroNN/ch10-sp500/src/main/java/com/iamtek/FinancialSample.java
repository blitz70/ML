package com.iamtek;

import java.text.DecimalFormat;
import java.util.Date;

import org.encog.util.csv.ReadCSV;

public class FinancialSample implements Comparable<FinancialSample> {

	private double amount;
	private double rate;
	private Date date;
	private double percent;
	
	@Override
	public int compareTo(FinancialSample other) {	//sort by date
		return getDate().compareTo(other.getDate());
	}

	@Override
	public String toString() {
		DecimalFormat fm = new DecimalFormat("##.####");
		StringBuilder result = new StringBuilder();
		
		result.append(ReadCSV.displayDate(this.date));
		
		result.append(", Amount: ");
		result.append(fm.format(amount));

		result.append(", Prime rate: ");
		result.append(rate);
		
		result.append(", Previous percent: ");
		result.append(fm.format(percent*100));
		
		return result.toString();
	}

	public double getAmount() {
		return amount;
	}

	public void setAmount(double amount) {
		this.amount = amount;
	}
	
	public double getRate() {
		return rate;
	}
	
	public void setRate(double rate) {
		this.rate = rate;
	}
	
	public Date getDate() {
		return date;
	}

	public void setDate(Date date) {
		this.date = date;
	}

	public double getPercent() {
		return percent;
	}

	public void setPercent(double percent) {
		this.percent = percent;
	}

}
