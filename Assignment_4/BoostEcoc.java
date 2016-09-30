package neu.ml.assignment4;


import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;


/**
 * Created by nikhilk on 11/5/15.
 */

class EcocFunction{
    HashSet<Integer> ones;
    HashSet<Integer> zeros;
    HashMap<Integer, Double> labelmap;

    public EcocFunction(){
        ones = new HashSet<>();
        zeros = new HashSet<>();
        labelmap = new HashMap<>();
    }
}

class Hypoth{
    ArrayList<DecisionStump.Stump> stlist = new ArrayList<>();
    ArrayList<Double> alphalist = new ArrayList<>();

    public Hypoth(){
        stlist = new ArrayList<>();
        alphalist = new ArrayList<>();

    }
}


public class BoostEcoc {
    public  HashMap<Integer, Double> trainmap;
    public double [][] train;

    public HashMap<Integer, Double> testmap;
    public double [][] test;

    public double [][] table;

    public ArrayList<EcocFunction> functions;
    public ArrayList<Hypoth> hypothesis;


    public void setData(double [][] train, double [] trlab, double [][] test , double [] telab){
        this.test = test;
        this.train = train;

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


    public void genData(){
        ParseEcoc parse = new ParseEcoc();

        HashMap<Integer, String> data =  parse.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_4/ECOC/8newsgroup/test.trec/feature_matrix.txt");
        parse.buildMat(data);
        test = parse.data;
        testmap = parse.labelmap;


        data = parse.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_4/ECOC/8newsgroup/train.trec/feature_matrix.txt");
        parse.buildMat(data);
        train = parse.data;
        trainmap = parse.labelmap;
    }


    public double[][] genEcocTable(int rows, int cols){
        table = new double[rows][cols];
        Random rand = new Random();

        for (int row = 0; row < table.length; row++){
            for (int col = 0; col < table[0].length; col++ ){
                int index = rand.nextInt(Integer.MAX_VALUE - 1);
                int val = 1;
                if (index % 2 == 0){
                    val = 0;
                }
                table[row][col] = val;
                System.out.print(val + " ");
            }
            System.out.print('\n');
        }
        return table;
    }


    public ArrayList<EcocFunction> genLabelFunctions(double [][] data, HashMap<Integer, Double> map){
        functions  = new ArrayList<>();

        for (int col = 0; col < data[0].length ; col ++){
            EcocFunction ef = new EcocFunction();

            // Iterate over row and put rownum in zeros and ones based on value
            for (int row =0 ; row < data.length ; row++){
                if (data[row][col] == 0){
                    ef.zeros.add(row);
                } else {
                    ef.ones.add(row);
                }
            }

            // Generate the labelmap for current function
            for (int row =0; row < map.size(); row++){
                double val = map.get(row);

                if (ef.zeros.contains((int) val)){
                    ef.labelmap.put(row,(double) 0);
                } else {
                    ef.labelmap.put(row, (double) 1);
                }
            }
            //System.out.println(ef.labelmap);
            // Add the object to list
            functions.add(ef);
        }

        return functions;
    }


    public void runBoosting() {
        SimpleMatrix trainmat = new SimpleMatrix(train);

        hypothesis = new ArrayList<>();

        for (int col = 0; col < functions.size(); col++) {
            Hypoth hp = new Hypoth();
            HashMap<Integer, Double> map = functions.get(col).labelmap;

            DecisionStump ds = new DecisionStump(map, trainmat);
            DecisionStump.Stump best;
            double alpha;

            int counter = 1;

            while (counter <= 2000) {
                Random rand = new Random();
                best = ds.getRandomStump(rand);
                //best = ds.getBestStump();
                alpha = ds.computeAlpha(best);
                hp.stlist.add(best);
                hp.alphalist.add(alpha);

                ds.updateDistribution(ds.distribution, alpha, best, trainmat);

                counter++;
                //System.out.println(counter);
            }
            System.out.println(hp.stlist.get(hp.stlist.size() - 1).rounderr);
            //System.out.println(hp.alphalist);
            hypothesis.add(hp);
        }
    }


    public double [] makePredictions(){
        double [] finalpredict = new double[test.length];

        for (int row = 0; row < test.length; row++){

            double [] predict = new double[functions.size()];
            for (int col = 0 ; col < functions.size(); col++){
                Hypoth hyp = hypothesis.get(col);
                EcocFunction ecf = functions.get(col);

                predict[col] = getPrediction(hyp.stlist, hyp.alphalist, test, row);
            }
           finalpredict[row] =  maxMatch(predict);
        }
        return finalpredict;
    }


    public double getPrediction(ArrayList<DecisionStump.Stump> stlist,
                              ArrayList<Double> alphalist, double [][] data, int row ){

        double prediction = 0;
        for (int itr=0; itr < stlist.size(); itr++){
            int feature = stlist.get(itr).feature;
            double threshold = stlist.get(itr).threshold;

            if (data[row][feature] >= threshold) {
                prediction = prediction + alphalist.get(itr);
            } else {
                prediction = prediction - alphalist.get(itr);
            }
        }
        if (prediction < 0) {
            return 0;
        } else {
            return  1;
        }
    }


    public double maxMatch(double [] predict){
        double label = -1;
        int match = Integer.MIN_VALUE;
        for (int row = 0; row < table.length; row++){

            int curmatch = 0;
            for(int col = 0; col < table[0].length; col++){
                if (table[row][col] == predict[col]){
                    curmatch++;
                }

            }

            if (curmatch > match){
                match = curmatch;
               label = row;
            }
        }
        return label;
    }


    public void accCalc(HashMap<Integer, Double> labelmap, double [] predictions){
            int match = 0;
            for (int itr = 0; itr < predictions.length ; itr++){
                if (predictions[itr] == labelmap.get(itr)){
                    match++;
                }
                System.out.println("Actual " + labelmap.get(itr) + " Prediction " + predictions[itr]);
            }
        double acc = (double) match / labelmap.size();
        System.out.print("Accuracy -> " + acc);
    }


    public static void main(String [] args){
        BoostEcoc be = new BoostEcoc();


        // Populate data, generate Ecoc table and generate labels per function
        be.genData();
        be.table =  be.genEcocTable(8, 20);
        be.genLabelFunctions(be.table, be.trainmap);

        //Perform boosting, make predictions,
        be.runBoosting();
        double [] predictions = be.makePredictions();

        be.accCalc(be.testmap, predictions);


        /**
        be.table = be.genEcocTable();
        double [] my = {0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1};
        System.out.println(be.maxMatch(my));
         **/

    }
}
