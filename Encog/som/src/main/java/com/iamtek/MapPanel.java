package com.iamtek;

import java.awt.Color;
import java.awt.Graphics;

import javax.swing.JPanel;

import org.encog.mathutil.matrices.Matrix;

public class MapPanel extends JPanel {

	private static final long serialVersionUID = 1L;
	
	private static final int CELL_SIZE = 8;
	public static final int ROWS = 50;
	public static final int COLS = 50;
	private Matrix weights;

	public MapPanel(SomColors som) {
		this.weights = som.getNetwork().getWeights();
	}

	private int convertColor(double color){
		// -1 = 0, +1 = 255
		double result = 255*(color + 1)/2;
		return (int)result;
	}
	
	public void paint(Graphics cell){
		for (int y = 0; y < ROWS; y++) {
			for (int x = 0; x < COLS; x++) {
				int index = (y*ROWS)+x;
				int red = convertColor(weights.get(index, 0));
				int green = convertColor(weights.get(index, 1));
				int blue = convertColor(weights.get(index, 2));
				cell.setColor(new Color(red, green, blue));
				cell.fillRect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE);
			}
		}
	}

}
