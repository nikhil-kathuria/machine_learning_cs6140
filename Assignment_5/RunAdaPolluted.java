package neu.ml.assignment5;

import neu.ml.assignment4.IterationStats;


import java.util.HashMap;

/**
 * Created by nikhilk on 11/10/15.
 */

public class RunAdaPolluted {

    public HashMap<Integer, Double> map(double [] label){
        HashMap<Integer, Double> mymap = new HashMap<>();

        for (int row=0; row < label.length; row++){
            mymap.put(row, label[row]);
        }

        return mymap;
    }

    public void runPolluted(){
        double [][] train;
        double [][] test;

        double [] testlabel;
        double [] trainlabel;



        ParseData pd = new ParseData();
        train = pd.populateData(pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/train_feature.txt"));
        test = pd.populateData(pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/test_feature.txt"));

        trainlabel = pd.populateLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/train_label.txt");
        testlabel = pd.populateLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/test_label.txt");


        HashMap<Integer, Double> testmap = map(testlabel);
        HashMap<Integer, Double> trainmap = map(trainlabel);

        System.out.println(testmap.size());
        System.out.println(trainmap.size());
        System.out.println(test.length);
        System.out.println(train.length);

        //IterationStats its = new IterationStats(new SimpleMatrix(train), trainmap, new SimpleMatrix(test), testmap);
        IterStats its = new IterStats(train, trainlabel, test, testlabel);
        its.runs =400;
        its.tillConvergence();
    }


    public static void main(String [] args){
        RunAdaPolluted rap = new RunAdaPolluted();
        rap.runPolluted();
    }
}
