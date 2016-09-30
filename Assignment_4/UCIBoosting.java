package neu.ml.assignment4;

import neu.ml.assignment1.CrossTrainData;
import neu.ml.assignment1.DecisionTree;
import neu.ml.assignment1.PredictClassification;
import org.ejml.simple.SimpleMatrix;

import java.util.*;

/**
 * Created by nikhilk on 11/1/15.
 */
public class UCIBoosting {
    public double [][] trainmat;
    public double [][] testmat;

    public HashMap<Integer, Double> testmap;
    public HashMap<Integer, Double> trainmap;

    ArrayList<DecisionStump.Stump> stlist = new ArrayList<>();
    ArrayList<Double> alphalist = new ArrayList<>();

    public int percentage;
    public String splitstring = ",";
    public int[] array = {5, 10, 15, 20, 30, 50, 80};



    public void buildTest(HashMap<Integer, String> data, HashMap<Integer, HashSet<Integer>> bucketmap,
                          int key) {
        HashMap<Integer, Double> labelmap = new HashMap();
        HashSet<Integer> rowset = bucketmap.get(key);
        int rows = rowset.size();

        int columns = data.get(0).split(splitstring).length;
        double[][] data2d = new double[rows][columns - 1];
        int rowid = 0;

        for (int row : rowset) {
            String[] datarow = data.get(row).split(splitstring);
            for(int col = 0; col < columns - 1; col++) {
                data2d[rowid][col] = Double.valueOf(datarow[col]);
            }
            labelmap.put(rowid, Double.valueOf(datarow[columns - 1]));
            rowid++;
        }
        testmap = labelmap;
        testmat = data2d;
    }



    public void buildTrain(HashMap<Integer, String> data, HashMap<Integer, HashSet<Integer>> bucketmap,
                           HashSet<Integer> set, int num){
        HashMap<Integer, Double> labelmap = new HashMap<>();

        HashSet<Integer> rowset = new HashSet<>();
        for(int key : set) {
            rowset.addAll(bucketmap.get(key));
        }

        ArrayList<Integer> mylist = new ArrayList<Integer>(rowset);
        Collections.shuffle(mylist);
        percentage = (mylist.size() * num) / 100;

        rowset = new HashSet<>();
        for (int itr = 0; itr < percentage; itr++){
            rowset.add(mylist.get(itr));
        }

        int rows = rowset.size();
        int columns = data.get(0).split(",").length;
        double[][] data2d = new double[rows][columns - 1];

        int rowid = 0;
        for (int row : rowset) {
            String[] datarow = data.get(row).split(",");
            for(int col = 0; col < columns - 1; col++) {
                data2d[rowid][col] = Double.valueOf(datarow[col]);
            }
            labelmap.put(rowid, Double.valueOf(datarow[columns - 1]));
            rowid++;
        }
        trainmap = labelmap;
        trainmat = data2d;
    }


    public void buildAndPredict(CrossTrainData ctd) {

        // Total buckets and tree holder
        HashSet<Integer> total = new HashSet<>(ctd.bucketmap.keySet());
        double summse = 0;
        double sumacc = 0;
        double acc = 0;
        double mse = 0;


        for (int key : total) {
            // Assign test and train. Update train
            HashSet<Integer> train = new HashSet<>(total);
            HashSet<Integer> test = new HashSet<>(key);
            train.remove(key);

            buildTest(ctd.datamap, ctd.bucketmap, key);
            for (int val : array){
                buildTrain(ctd.datamap, ctd.bucketmap, train, val);
                //System.out.println(trainmap.size());
                //System.out.println(testmap.size());

                //System.out.println(trainmat.length + " " + testmat.length);
                //System.exit(0);

                SimpleMatrix trmat = new SimpleMatrix(trainmat);
                SimpleMatrix temat = new SimpleMatrix(testmat);

                IterationStats its = new IterationStats(trmat, trainmap, temat, testmap);
                its.runs = 300;
                its.tillConvergence();
            }
            System.exit(0);
        }
    }


    public void predictCrossValidation(CrossTrainData ctd){
        HashSet<Integer> total = new HashSet<>(ctd.bucketmap.keySet());
        double summse = 0;
        double sumacc = 0;
        double acc = 0;
        double mse = 0;


        for (int key : total) {
            // Assign test and train. Update train
            HashSet<Integer> train = new HashSet<>(total);
            HashSet<Integer> test = new HashSet<>(key);
            test.add(key);
            train.remove(key);


            PredictClassification prc = new PredictClassification();
            prc.buildMatrix(ctd.datamap, ctd.bucketmap, train, true);
            prc.buildMatrix(ctd.datamap, ctd.bucketmap, test, false);

            IterationStats its = new IterationStats(prc.trainmat, prc.trainmap, prc.testmat, prc.testmap);
            its.runs = 220;
            its.tillConvergence();
            System.exit(0);
        }

    }

    public static void main(String [] args) {
        UCIBoosting ucb = new UCIBoosting();

        CrossTrainData crx = new CrossTrainData();
        crx.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_4/UCI/crx/crx_mm.data");
        crx.genBuckets(new HashSet<Integer>(crx.datamap.keySet()));


        //ucb.predictCrossValidation(crx);
        //ucb.buildAndPredict(crx);

        CrossTrainData vote = new CrossTrainData();
        vote.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_4/UCI/vote/vote_mm.data");
        vote.genBuckets(new HashSet<Integer>(vote.datamap.keySet()));

        //ucb.predictCrossValidation(vote);
        ucb.buildAndPredict(crx);
    }
}
