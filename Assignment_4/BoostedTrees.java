package neu.ml.assignment4;

import neu.ml.assignment1.*;

import java.util.HashMap;
import java.util.HashSet;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by nikhilk on 11/2/15.
 */


public class BoostedTrees {
    HashMap<Integer, Double> labelmap;
    HashMap<Integer, Double> testmap;
    HashMap<Integer, Double> trainmap;

    SimpleMatrix testmat;
    SimpleMatrix trainmat;


    public void updateprediction(double [] finalprediction ,double [] predictions){
        for (int itr = 0; itr < finalprediction.length; itr++){
            finalprediction[itr] += predictions[itr];
        }
    }


    public void updatelabels(double [] predictions, HashMap<Integer, Double> map){
        for (int itr =0 ; itr < predictions.length; itr++){
            double val = map.get(itr) - predictions[itr];
            map.put(itr, val);
        }
    }


    public void mseCalc(HashMap<Integer, Double> labelmap, double [] labels){
        double mse = 0;

        for (int itr= 0 ; itr < labels.length ; itr++) {
            System.out.println(labelmap.get(itr) + " " + labels[itr]);
            mse += Math.pow( (labelmap.get(itr) - labels[itr]), 2);
        }

        mse = mse / labels.length;

        System.out.println("MSE " + mse );
    }



    public void performBoosting(){
        int counter = 1;

        PredictRegression pr = new PredictRegression();
        double [] testpredict = new double[testmap.size()];
        double [] trainpredict = new double[trainmap.size()];


        double [] finaltest = new double[testmap.size()];
        double [] finaltrain = new double[trainmap.size()];



        while (counter <= 10) {
            DecisionTree head = new DecisionTree(new HashSet<>(trainmap.keySet()), labelmap, trainmat, true);
            head.buildTree();

            testpredict = pr.predictClass(head, testmat);
            trainpredict = pr.predictClass(head, trainmat);

            updateprediction(finaltest, testpredict);
            updateprediction(finaltrain, trainpredict);

            updatelabels(trainpredict, labelmap);
            counter++;

        }

        System.out.println("Printing MSE for test");
        mseCalc(testmap, finaltest);

        System.out.println("Printing MSE for Train");
        mseCalc(trainmap, finaltrain);
    }


    public static void main(String [] args){
        ParserAndBuildMatrix pab = new ParserAndBuildMatrix();
        HashMap<Integer, String> data =
                pab.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/housing_test.txt");
        pab.buildMatrix(data);

        BoostedTrees bt = new BoostedTrees();

        bt.testmap = pab.labelmap;
        bt.testmat = pab.matrix;


        data = pab.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/housing_train.txt");
        pab.buildMatrix(data);

        bt.trainmap = pab.labelmap;
        bt.trainmat = pab.matrix;
        bt.labelmap = new HashMap<>(bt.trainmap);

        bt.performBoosting();


    }

}
