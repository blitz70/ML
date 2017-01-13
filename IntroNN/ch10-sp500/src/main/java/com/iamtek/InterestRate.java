package com.iamtek;

import java.util.Date;

public class InterestRate implements Comparable<InterestRate>{

	private Date date;
	private double rate;
	
	@Override
	public int compareTo(InterestRate other) {	//sort by date
		return getDate().compareTo(other.getDate());
	}

	public InterestRate(Date date, double rate) {
		this.date = date;
		this.rate = rate;
	}

	public Date getDate() {
		return date;
	}

	public void setDate(Date date) {
		this.date = date;
	}

	public double getRate() {
		return rate;
	}

	public void setRate(double rate) {
		this.rate = rate;
	}

}
