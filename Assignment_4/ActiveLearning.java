package neu.ml.assignment4;

import java.util.*;
import neu.ml.assignment1.PredictClassification;
import org.ejml.simple.SimpleMatrix;

/**
 * Created by nikhilk on 11/2/15.
 */

public class ActiveLearning {
    HashSet<Integer> trainset;
    HashSet<Integer> testset;

    public HashMap<Integer, Double> testmap;
    public HashMap<Integer, Double> trainmap;

    SimpleMatrix trainmat;
    SimpleMatrix testmat;

    ArrayList<DecisionStump.Stump> stlist = new ArrayList<>();
    ArrayList<Double> alphalist = new ArrayList<>();

    int start = 5;
    int incrementsize = 2 ;
    int size;
    int increment;


    public void initset(Set<Integer> set, ParseSpam pm) {
        trainset = new HashSet<>();
        testset = new HashSet<>();
        int init = (size * start) / 100;

        ArrayList<Integer> mylist = new ArrayList<>(set);
        Collections.shuffle(mylist);

        for (int itr = 0; itr < mylist.size(); itr++) {
            if (itr <= init) {
                trainset.add(mylist.get(itr));
            } else {
                testset.add(mylist.get(itr));
            }
        }
        increment = (2 * size) / 100;
        genMatMap(pm.data, pm.labelmap);
    }


    public void genMatMap(double[][] data, HashMap<Integer, Double> mymap) {
        double[][] trainmat = new double[trainset.size()][data[0].length];
        double[][] testmat = new double[testset.size()][data[0].length];

        testmap = new HashMap<>();
        trainmap = new HashMap<>();

        //System.out.println("Train row " + trainmat.length + " Train col " + trainmat[0].length );
        //System.out.println("Test row " + testmat.length + " Test col " + testmat[0].length );
        //System.out.println("data row " + data.length + " data col " + data[0].length );

        int counter = 0;
        for (int row : trainset) {
            for (int col = 0; col < data[0].length; col++) {
                trainmat[counter][col] = data[row][col];
                trainmap.put(counter, mymap.get(row));
            }
            counter++;
        }

        counter = 0;
        for (int row : testset) {
            for (int col = 0; col < data[0].length; col++) {
                testmat[counter][col] = data[row][col];
                testmap.put(counter, mymap.get(row));
            }
            counter++;
        }

        this.testmat = new SimpleMatrix(testmat);
        this.trainmat = new SimpleMatrix(trainmat);
    }


    public double [] compHypoth(SimpleMatrix mat) {
        double[] predictions = new double[mat.numRows()];

        for (int row = 0; row < mat.numRows(); row++) {
            double prediction = 0;

            for (int itr = 0; itr < stlist.size(); itr++) {
                int feature = stlist.get(itr).feature;
                double threshold = stlist.get(itr).threshold;

                if (mat.get(row, feature) >= threshold) {
                    prediction = prediction + alphalist.get(itr);
                } else {
                    prediction = prediction - alphalist.get(itr);
                }
            }
            predictions[row] = prediction;
        }
        return predictions;
    }


    public void updateSets(double [] predictions){
        HashMap<Integer, Double> mymap = new HashMap<>();

        int counter = 0;
        for (int row : testset) {
            mymap.put(row, Math.abs(predictions[counter]));
            //mymap.put(row, predictions[counter]);
            counter++;
        }

        // Sort the map
        List<Map.Entry<Integer, Double>> myList = new ArrayList<Map.Entry<Integer, Double>>(mymap.entrySet());
        Collections.sort(myList, new Comparator<Map.Entry<Integer, Double>>() {
            @Override
            public int compare(Map.Entry<Integer, Double> obj1, Map.Entry<Integer, Double> obj2) {
                return obj1.getValue().compareTo(obj2.getValue());
            }
        });


        HashSet<Integer> myset = new HashSet<>();
        for (int itr =0; itr < increment; itr++){
            myset.add(myList.get(itr).getKey());
        }

        // Update the test and train set
        testset.removeAll(myset);
        trainset.addAll(myset);

    }


    public void randomSets(){
        ArrayList<Integer> myList = new ArrayList<>(testset);
        Collections.shuffle(myList);

        HashSet<Integer> myset = new HashSet<>();
        for (int itr =0; itr < increment; itr++){
            myset.add(myList.get(itr));
        }

        // Update the test and train set
        testset.removeAll(myset);
        trainset.addAll(myset);

    }


    public double averageError(double [] predictions, HashMap<Integer, Double> map) {
        int[] results = new int[predictions.length];

        for (int row = 0; row < predictions.length; row++) {
            // Return the average Error i.e.
            if (predictions[row] < 0) {
                results[row] = 0;
            } else {
                results[row] = 1;
            }
        }

        int error = 0;
        for (int itr = 0; itr < results.length; itr++) {
            if (results[itr] != map.get(itr)) {
                error++;
            }
        }
        return (double) error / results.length;
    }



    public void runActiveLearning(boolean random) {
        ParseSpam pm = new ParseSpam();
        this.size = pm.labelmap.size();
        initset(pm.labelmap.keySet(), pm);

        while (trainset.size() < (pm.labelmap.size() / 2)) {
            DecisionStump ds = new DecisionStump(trainmap, trainmat);
            DecisionStump.Stump best;
            double alpha;

            stlist = new ArrayList<>();
            alphalist = new ArrayList<>();
            double [] mytrain = new double[trainmap.size()];
            double [] mytest = new double[testmap.size()];

            int counter = 1;
            while (counter <= 8) {
                Random rand = new Random();
                best = ds.getRandomStump(rand);
                best = ds.getBestStump();
                alpha = ds.computeAlpha(best);



                stlist.add(best);
                alphalist.add(alpha);

                ds.updateDistribution(ds.distribution, alpha, best, trainmat);
                //mytest = compHypoth(testmat);
                //mytrain = compHypoth(trainmat);
                //double trainnerr = averageError(myscore, trainmap);
                counter++;
            }
            mytest = compHypoth(testmat);
            double testerr = averageError(mytest, testmap);

            double cursize = ((double) trainset.size() / pm.labelmap.size()) * 100;

            System.out.println("Test Error at " + cursize +" percent of data " + testerr);

            if (random) {
                randomSets();
            } else {
                updateSets(mytest);
            }
            genMatMap(pm.data, pm.labelmap);

        }
    }


    public static void main(String [] args){
        ActiveLearning al = new ActiveLearning();
        al.runActiveLearning(false);
    }




}
