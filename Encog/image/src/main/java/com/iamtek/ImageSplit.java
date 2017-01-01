package com.iamtek;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

//http://www.vision.ime.usp.br/~daniel/sibgrapi2005/

public class ImageSplit {

	private static final int COLS = 28;
	private static final int ROWS = 24;
	private static final int OFFSET_X = 0;
	private static final int OFFSET_Y = 0;
	
	public static int split(String sourcPath, String sourceName, String targetPath, String targetName, int startno){
		File file = new File(sourcPath, sourceName);
		BufferedImage image = null;
		System.out.print("Getting image:" + sourceName + ", ");
		try {
			image = ImageIO.read(file);
			System.out.print("done. ");
		} catch (IOException e) {
			System.out.print("error. ");
			e.printStackTrace();
		}
		int numberChunk = COLS*ROWS;
		int widthChunk = image.getWidth()/COLS;
		int heightChunk = image.getHeight()/ROWS;
		BufferedImage[] images = new BufferedImage[numberChunk];
		int readCount = 0; 
		System.out.print("Splitting, ");
		for (int y = 0; y < ROWS; y++) {
			for (int x = 0; x < COLS; x++) {
				images[readCount] = new BufferedImage(widthChunk, heightChunk, image.getType());
				Graphics2D gr = images[readCount].createGraphics();
				gr.drawImage(image,
						0, 0, widthChunk, heightChunk,
						widthChunk*x+OFFSET_X, heightChunk*y+OFFSET_Y, widthChunk*(x+1)+OFFSET_X, heightChunk*(y+1)+OFFSET_Y,
						null);
				gr.dispose();
				readCount++;
			}
		}
		System.out.print("done. Writing, ");
		int writeCount = 0;
		try {
			for (int i = 0; i < images.length; i++) {
				ImageIO.write(images[i], "jpg", new File(targetPath, targetName+(i+startno)+".jpg"));
				writeCount = i+1;
			}
			System.out.println(readCount +" images done.");
		} catch (IOException e) {
			System.out.println("error.");
			e.printStackTrace();
		}
		
		return (writeCount-1 +startno);
		
	}
	
	public static void main(String[] args) {
		
		//training
		int count=0;
		String path = "d:/code/digits/training/";
		for (int j = 0; j < 10; j++) {
			for (int i = 1; i < 8; i++) {
				count = split(path+j+"/", "digits"+j+"-"+i+".png", path, "digits"+j+"-", count+i);
			}
			count=0;
		}

		//test
		count=0;
		path = "d:/code/digits/test/";
		for (int j = 0; j < 10; j++) {
			for (int i = 1; i < 4; i++) {
				count = split(path+j+"/", "digits"+j+"-"+(i+7)+".png", path, "digits"+j+"-", count+i);
			}
			count=0;
		}
	}
}
