package neu.ml.assignment6;

import neu.ml.assignment5.ParseData;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by nikhilk on 12/19/15.
 */

public class SMODigitsOne {
    double [][] train;
    double [][] test;
    double [] testlabels;
    double [] trainlables;

    double [] labelsTrain;
    double [][] trainData;

    HashMap<Integer, HashSet<Integer>> rowmap;
    double [][] confidenceTable;


    public void initialize(){
        ParseData pd = new ParseData();
        trainlables =  pd.populateLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/s1labels.txt");
        train = pd.populateData(pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/s1train.txt"));

        testlabels = pd.populateLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htestlabels.txt");

        test = pd.populateData(pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htest.txt"));
    }


    public void populateRowMap(){
        rowmap = new HashMap<>();
        int label;
        for (int row = 0 ; row < trainlables.length; row++){
            label = (int) trainlables[row];

            if (rowmap.containsKey(label)) {
                rowmap.get(label).add(row);
            }
            else{
                rowmap.put(label, new HashSet<Integer>(row));
            }
        }


    }


    public void formData(int i, int j){
        int sizei = rowmap.get(i).size();
        int sizej = rowmap.get(j).size();

        System.out.println(sizei + " " + sizej);

        labelsTrain = new double[sizei + sizej];
        trainData = new double[sizei + sizej][train[0].length];

        System.out.println("Train Data " + trainData.length + " " + trainData[0].length);
        System.out.println("Labels Length " + labelsTrain.length);

        int counter = 0;

        for (int row = 0; row < trainlables.length; row ++){
            if (trainlables[row] == i){
                labelsTrain[counter] = 1;
                genData(counter, row);
                counter ++;

            } else if (trainlables[row] == j){
                labelsTrain[counter] = -1;
                genData(counter, row);
                counter ++;
            }

            System.out.println("Row " + trainlables[row]);
        }



    }


    public void genData(int counter, int idx ){
        for (int col=0; col < train[0].length ; col++){
            trainData[counter][col] = train[idx][col];
        }
    }


    public void updateTable(int col, int i, int j, double[] weights, double B){
        for (int row = 0; row < confidenceTable.length; row++){
            double Fx=0;

            // Get Fx = W.T * X
            for (int itr=0; itr < weights.length; itr++){
                Fx += weights[itr] * test[row][col];
            }

            Fx = Fx - B;

            if (Fx > 0){
                confidenceTable[row][col] = i;
            } else {
                confidenceTable[row][col] = j;
            }

        }

    }


    public void getAccuracy(){
        int hit=0;

        for (int row =0 ;row < confidenceTable.length; row++){

            HashMap<Double ,Integer > mymap = new HashMap<>();

            for(int col =0 ;col < confidenceTable[0].length ; col++){
                double prediction = confidenceTable[row][col];

                if (mymap.containsKey(prediction)){
                    mymap.put(prediction, mymap.get(prediction) +  1);
                } else {
                    mymap.put(prediction, 1);
                }
            }

            int val = Integer.MIN_VALUE;
            double indx = 1;


            for (double key : mymap.keySet()){
                if (mymap.get(key) > val){
                    val = mymap.get(key);
                    indx =key;
                }
            }

            if (trainlables[row] == indx){
                hit++;
            }

        }

        System.out.println("Accuracy -> " + (double) hit / trainlables.length);

    }


    public void trainAll(){
        Normalize nm = new Normalize();
        train = nm.normalize(train);
        test = nm.normalize(test);


        int end = (rowmap.size() * (rowmap.size() -1)) / 2;  // N Choose 2
        confidenceTable = new double [testlabels.length][end];


        int  counter = 0;

        for(int slow = 0 ; slow <= end; slow++){
            for (int fast=slow + 1 ; fast <=end ;fast++){

                formData(slow, fast);

                //System.out.print("Gen");//Arrays.deepToString(trainData));
                //System.out.print(Arrays.toString(labelsTrain));


                SMOSolver sms = new SMOSolver(trainData, labelsTrain);
                sms.Solver();

                updateTable(counter, slow, fast, sms.weights, sms.B);

                System.out.println("Done with Model Number " + counter);

                counter++;

            }

        }


    }

    public static void main(String [] args){
        SMODigitsOne smo = new SMODigitsOne();
        smo.initialize();
        smo.populateRowMap();
        smo.trainAll();
        smo.getAccuracy();
    }

}
