package com.iamtek;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class ImageSplit {

	private static final String FILE_PATH = "src/main/resources/";
	private static final String FILE_NAME = "digits0-1.png";
	private static final int COLS = 28;	//x28
	private static final int ROWS = 24;	//y24
	private static final int OFFSET_X = 0;
	private static final int OFFSET_Y = 10;
	
	
	public static void main(String[] args) {
		
		File file = new File(FILE_PATH, FILE_NAME);
		//FileInputStream fis = new FileInputStream(file);
		BufferedImage image = null;
		try {
			image = ImageIO.read(file);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		int numberChunk = COLS*ROWS;
		int widthChunk = image.getWidth()/COLS;
		int heightChunk = image.getHeight()/ROWS;
		BufferedImage[] images = new BufferedImage[numberChunk];
		int count = 0; 
		for (int y = 0; y < ROWS; y++) {
			for (int x = 0; x < COLS; x++) {
				images[count] = new BufferedImage(widthChunk, heightChunk, image.getType());
				Graphics2D gr = images[count].createGraphics();
				gr.drawImage(image,
						0, 0, widthChunk, heightChunk,
						widthChunk*x+OFFSET_X, heightChunk*y+OFFSET_Y, widthChunk*(x+1)+OFFSET_X, heightChunk*(y+1)+OFFSET_Y,
						null);
				gr.dispose();
				count++;
			}
		}
		System.out.println("Splitting done");
		System.out.println("writing...");
		for (int i = 0; i < images.length; i++) {
			try {
				ImageIO.write(images[i], "jpg", new File(FILE_PATH,(i+1)+".jpg"));
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

}
