package com.iamtek.timeseries_stock;

import java.io.File;
import java.net.URISyntaxException;

public class Main {

	public static void main(String[] args) {
		
		File dataFile = null;
		StockPredict predict = new StockPredict();

		try {
			System.out.println("Kospi");
			dataFile = new File(StockPredict.class.getResource("/kospi.data").toURI());
			predict.run(dataFile);
			System.out.println("\nSamsungE");
			dataFile = new File(StockPredict.class.getResource("/samsunge.data").toURI());
			predict.run(dataFile);
			System.out.println("\nNCsoft");
			dataFile = new File(StockPredict.class.getResource("/ncsoft.data").toURI());
			predict.run(dataFile);
		} catch (URISyntaxException e) {
			e.printStackTrace();
		}

	}

}
