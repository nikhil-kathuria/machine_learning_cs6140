package neu.ml.assignment5;

import  neu.ml.assignment4.BoostEcoc;

import  neu.ml.assignment5.EcocBoosting;

import java.util.HashMap;

/**
 * Created by nikhilk on 11/18/15.
 */

public class RunMINST {
    double [][] train;
    double [][] test;

    double [] trainlabels;
    double [] testlabels;

    HashMap<Integer, Double> trainmap;
    HashMap<Integer, Double> testmap;


    public void populateData(){
        ParseData pd = new ParseData();
        train = pd.populateData(pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htrain.txt"));
        test = pd.populateData(pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htest.txt"));

        trainlabels = pd.populateLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htrainlabels.txt");
        testlabels = pd.populateLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htestlabels.txt");
    }


    public void setData(double [] trlab, double [] telab){

        HashMap<Integer, Double> mymap = new HashMap<>();
        for (int itr=0; itr < trlab.length; itr++){
            mymap.put(itr, trlab[itr]);
        }

        trainmap = mymap;

        mymap = new HashMap<>();
        for (int itr=0; itr < telab.length; itr++){
            mymap.put(itr, telab[itr]);
        }

        testmap = mymap;

    }


    public void oldDecision(){
        BoostEcoc be = new BoostEcoc();

        // Populate data, generate Ecoc table and generate labels per function
        be.test = this.test;
        be.train = this.train;
        be.trainmap = this.trainmap;
        be.testmap = this.testmap;

        be.table =  be.genEcocTable(10, 50);
        be.genLabelFunctions(be.table, be.trainmap);

        //Perform boosting, make predictions,
        be.runBoosting();
        double [] predictions = be.makePredictions();

        be.accCalc(be.testmap, predictions);

    }

    public void newDecision(){
        EcocBoosting eb = new EcocBoosting();

        // Set data
        eb.setData(train, trainlabels, test, testlabels, 200);

        // Generate Ecoc table
        eb.table =  eb.genEcocTable(10, 50);
        eb.genLabelFunctions(eb.table, eb.trainlabels);

        //Perform boosting, make predictions,
        eb.runBoosting();
        double [] predictions = eb.makePredictions();

        eb.accCalc(eb.testlabels, predictions);
    }


    public static void main(String args []){
        RunMINST rm = new RunMINST();
        rm.populateData();
        rm.setData(rm.trainlabels, rm.testlabels);

        rm.oldDecision();
        //rm.newDecision();
    }
}
