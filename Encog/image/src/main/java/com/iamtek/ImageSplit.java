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
		
		int count=0;
		String sourcePath="";
		String targetPath="";
		
		//training
		sourcePath = "src/main/resources/digits/training/0/";
		targetPath = "src/main/resources/training/";
		for (int i = 1; i < 8; i++) {
			count = split(sourcePath, "digits0-"+i+".png", targetPath, "digits0-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/training/1/";
		for (int i = 1; i < 8; i++) {
			count = split(sourcePath, "digits1-"+i+".png", targetPath, "digits1-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/training/2/";
		for (int i = 1; i < 8; i++) {
			count = split(sourcePath, "digits2-"+i+".png", targetPath, "digits2-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/training/3/";
		for (int i = 1; i < 8; i++) {
			count = split(sourcePath, "digits3-"+i+".png", targetPath, "digits3-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/training/4/";
		for (int i = 1; i < 8; i++) {
			count = split(sourcePath, "digits4-"+i+".png", targetPath, "digits4-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/training/5/";
		for (int i = 1; i < 8; i++) {
			count = split(sourcePath, "digits5-"+i+".png", targetPath, "digits5-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/training/6/";
		for (int i = 1; i < 8; i++) {
			count = split(sourcePath, "digits6-"+i+".png", targetPath, "digits6-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/training/7/";
		for (int i = 1; i < 8; i++) {
			count = split(sourcePath, "digits7-"+i+".png", targetPath, "digits7-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/training/8/";
		for (int i = 1; i < 8; i++) {
			count = split(sourcePath, "digits8-"+i+".png", targetPath, "digits8-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/training/9/";
		for (int i = 1; i < 8; i++) {
			count = split(sourcePath, "digits9-"+i+".png", targetPath, "digits9-", count+i);
		}

		//test
		count=0;
		sourcePath = "src/main/resources/digits/test/0/";
		targetPath = "src/main/resources/test/";
		for (int i = 1; i < 4; i++) {
			count = split(sourcePath, "digits0-"+(i+7)+".png", targetPath, "digits0-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/test/1/";
		for (int i = 1; i < 4; i++) {
			count = split(sourcePath, "digits1-"+(i+7)+".png", targetPath, "digits1-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/test/2/";
		for (int i = 1; i < 4; i++) {
			count = split(sourcePath, "digits2-"+(i+7)+".png", targetPath, "digits2-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/test/3/";
		for (int i = 1; i < 4; i++) {
			count = split(sourcePath, "digits3-"+(i+7)+".png", targetPath, "digits3-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/test/4/";
		for (int i = 1; i < 4; i++) {
			count = split(sourcePath, "digits4-"+(i+7)+".png", targetPath, "digits4-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/test/5/";
		for (int i = 1; i < 4; i++) {
			count = split(sourcePath, "digits5-"+(i+7)+".png", targetPath, "digits5-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/test/6/";
		for (int i = 1; i < 4; i++) {
			count = split(sourcePath, "digits6-"+(i+7)+".png", targetPath, "digits6-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/test/7/";
		for (int i = 1; i < 4; i++) {
			count = split(sourcePath, "digits7-"+(i+7)+".png", targetPath, "digits7-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/test/8/";
		for (int i = 1; i < 4; i++) {
			count = split(sourcePath, "digits8-"+(i+7)+".png", targetPath, "digits8-", count+i);
		}
		count=0;
		sourcePath = "src/main/resources/digits/test/9/";
		for (int i = 1; i < 4; i++) {
			count = split(sourcePath, "digits9-"+(i+7)+".png", targetPath, "digits9-", count+i);
		}
		
	}

}
