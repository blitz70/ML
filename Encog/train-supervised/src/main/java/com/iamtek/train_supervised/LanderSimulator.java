package com.iamtek.train_supervised;

import java.text.DecimalFormat;

public class LanderSimulator {
	
	public static final double GRAVITY = 1.62;	//gravity of moon
	public static final double THRUST = 10;	//thrust of lunar lander
	public static final double TERMINAL_VELOCITY = 40;	//moon's terminal velocity (moon with air?)
	
	//lander's fuel remaining (L), time of flight (sec), altitude (m), velocity (m/s, + up, - down)
	private int fuel;
	private int seconds;
	private double altitude;
	private double velocity;
	
	//initialize
	public LanderSimulator() {
		this.fuel = 200;
		this.seconds = 0;
		this.altitude = 10000;
		this.velocity = 0;

	}
	
	public void turn(boolean thrust){
		//time
		seconds++;
		//velocity
		velocity-= GRAVITY;
		if(thrust && fuel>0){
			fuel--;
			velocity+= THRUST;
		}
		velocity = Math.max(-TERMINAL_VELOCITY, velocity);	//max down velocity
		velocity = Math.min(TERMINAL_VELOCITY, velocity);	//max up
		//altitude
		altitude+= velocity;
		if(altitude<0){
			altitude=0;
		}
	}
	
	public int score(){
		int score = (int) (fuel*10 + velocity*1000 + seconds);
		return score;
	}
	
	public boolean isFly(){
		boolean fly = false;
		if(altitude>0){
			fly = true;
		}
		return fly;
	}

	public StringBuilder telemtry() {
		DecimalFormat df = new DecimalFormat("#.##");
		StringBuilder result = new StringBuilder();
		result.append("Elapsed: " + seconds);
		result.append(" s, Fuel: " + fuel);
		result.append(" l, Velocity: " + df.format(velocity));
		result.append(" m/s, Altitude: " + (int)altitude);
		result.append(" m");
		return result;
	}

	public int getFuel() {
		return fuel;
	}

	public void setFuel(int fuel) {
		this.fuel = fuel;
	}

	public int getSeconds() {
		return seconds;
	}

	public void setSeconds(int seconds) {
		this.seconds = seconds;
	}

	public double getAltitude() {
		return altitude;
	}

	public void setAltitude(double altitude) {
		this.altitude = altitude;
	}

	public double getVelocity() {
		return velocity;
	}

	public void setVelocity(double velocity) {
		this.velocity = velocity;
	}

}
