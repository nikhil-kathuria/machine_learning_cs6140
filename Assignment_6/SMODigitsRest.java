package neu.ml.assignment6;

/**
 * Created by nikhilk on 12/18/15.
 */

import  neu.ml.assignment5.*;


public class SMODigitsRest {
    double [][] train;
    double [][] test;
    double [] testlabels;
    double [] trainlables;

    double [] labelsTrain;
    double [][] confidenceTable;


    public double[] convertLabels(double [] source ,int num){
        double [] labels = new double[source.length];

        for (int row = 0; row < source.length ; row++){
            if (source[row] == num){
                labels[row] = 1;
            } else {
                labels[row] = -1;
            }
        }
        return  labels;
    }


    public void initialize(){
        ParseData pd = new ParseData();
        trainlables =  pd.populateLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/s10labels.txt");
        train = pd.populateData(pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/s10train.txt"));

        testlabels = pd.populateLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htestlabels.txt");

        test = pd.populateData(pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htest.txt"));
    }



    public void updateTable(int col, double [] weights, double B){
        for (int row =0; row < testlabels.length ;row++) {
            double Fx = 0;

            for (int itr = 0; itr < weights.length; itr++) {
                Fx +=  test[row][itr] * weights[itr];
            }
            confidenceTable[row][col] = Fx - B ;
        }
    }


    public int dataClass(int row){
        double mymax = Double.NEGATIVE_INFINITY;
        int myindx = 1;

        for (int col= 0; col < confidenceTable[0].length ;col++){
            if (confidenceTable[row][col] > mymax ){
                mymax = confidenceTable[row][col];
                myindx = col;
            }
        }

        return myindx;
    }


    public void finalAcc(){
        int hit = 0;

        for (int row = 0; row < testlabels.length ; row++){
            double val = dataClass(row);
            if (val == testlabels[row]){
                hit++;
            }

        }

        System.out.println("Accuracy -> " + (double) hit / testlabels.length);
    }



    public void runDigits(){
        Normalize nm = new Normalize();
        train = nm.normalizeall(train);
        test = nm.normalizeall(test);

        confidenceTable = new double[test.length][10];


        for (int itr=0 ; itr < 10; itr++){
            labelsTrain = convertLabels(trainlables, itr);

            SMOSolver sms = new SMOSolver(train, labelsTrain);
            sms.Solver();

            updateTable(itr, sms.weights, sms.B);

            System.out.println("Done Class " + itr);
        }

    }



    public static void main(String [] args){
        SMODigitsRest sdr = new SMODigitsRest();
        sdr.initialize();
        sdr.runDigits();
        sdr.finalAcc();
    }
}
