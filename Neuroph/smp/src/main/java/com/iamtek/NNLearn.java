package com.iamtek;

import java.io.IOException;

/**
 * Created by Administrator on 2016-11-15.
 */
public class NNLearn {

    public static void main(String[] args) throws IOException {

        NNStockPredictor predictor = new NNStockPredictor(5, "C:\\Users\\Administrator\\Desktop\\CODE\\SPP\\src\\main\\resources\\rawTrainingData-kospi.csv");
        predictor.prepareData();

        System.out.println("Training starting");
        predictor.trainNetwork();

    }

}
