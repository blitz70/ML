package com.iamtek;

import java.io.IOException;

public class NNTest {

    public static void main(String[]args) throws IOException {

        NNStockPredictor predictor = new NNStockPredictor(5, "C:\\Users\\Administrator\\Desktop\\CODE\\SPP\\src\\main\\resources\\rawTrainingData-kospi.csv");

        System.out.println("Testing network");
        predictor.testNetwork();

    }

}
