package neu.ml.assignment1;

//import java.util.*;

/**
 * Created by nikhilk on 9/13/15.
 */


import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
//import java.util.Set;
import java.util.Map.Entry;
import java.util.List;
import java.util.Collections;
import java.util.ArrayList;
import java.util.Comparator;



import org.ejml.simple.SimpleMatrix;

public class DecisionTree {
    HashSet<Integer> dataset;
    DecisionTree left;
    DecisionTree right;

    boolean regression;
    double threshold;
    double minerror;
    int feature;
    double prediction;


    public final HashMap<Integer, Double> labelmap;
    public final SimpleMatrix mat;


    // Initialize the constructor
    public DecisionTree(HashSet<Integer> dataset,HashMap<Integer, Double> map, SimpleMatrix mat, boolean bool){
        this.dataset = dataset;
        labelmap = map;
        this.mat = mat;
        regression = bool;

        // Allocate prediction based on regression or classification model
        if (regression) {
            this.prediction = average(dataset, labelmap);
        } else {
            this.prediction = majority(dataset, labelmap);
        }

    }

    // Class to hold new split information
    public class SplitData{
        HashSet<Integer> left;
        HashSet<Integer> right;
        double error;
        int feature;
        double threshold;


        SplitData(double err){
            error = err;
        }

    }

    public DecisionTree startBuild(ParserAndBuildMatrix pab, Boolean bool) {
        DecisionTree head = new DecisionTree(new HashSet<>(pab.labelmap.keySet()), pab.labelmap, pab.matrix, bool);
        head.buildTree();
        return head;
    }

    public void buildTree() {
        // Dont proceed when we dataset is less than X percent of data
        if (dataset.size() < .2 * labelmap.size()){
            return;
        }

        SplitData best = new SplitData(Double.MAX_VALUE);
        SplitData current = new SplitData(Double.MAX_VALUE);

        //System.out.println("DataSet : " + dataset);

        for (int col=0; col < mat.numCols(); col++){
            HashMap<Integer, Double> map = new HashMap<>();
            for (int row : dataset ) {
                map.put(row, mat.get(row, col));
            }
            // Now we have map of dataset to feature values. computeCost now.
            current = computeBestCost(map);

            if(current.error < best.error) {
                best = current;
                best.feature = col;
            }
            //System.out.println("Current Error : " + current.error + " Best Error : " + best.error);
            //System.out.println("Current Feature : " + current.feature + " Best Feature : " + best.feature);
            //System.out.println("Entropy : " + best.error);
        }

        //System.out.println("Feature " + best.feature);
        //System.out.println("Predication " + this.prediction );
        //System.out.println("Left  " + best.left.size());
        //System.out.println("Right  " + best.right.size());
        //System.out.println("Error  " + best.error);
        // Allocate the the feature ID and value for the feature on which split is performed.
        this.feature = best.feature;
        this.threshold = best.threshold;
        this.minerror = best.error;


        if (best.left.size() > 0 && best.right.size() > 0) {
            left = new DecisionTree(best.left, labelmap, mat, regression);
            right = new DecisionTree(best.right, labelmap, mat, regression);

            left.buildTree();
            right.buildTree();
        }
    }


    protected SplitData computeBestCost(HashMap<Integer, Double> map) {

        // Sort the Map based on values
        List<Map.Entry<Integer, Double>> myList = new ArrayList<Map.Entry<Integer, Double>>(map.entrySet());
        Collections.sort(myList, new Comparator<Map.Entry<Integer, Double>>() {
            @Override
            public int compare(Entry<Integer, Double> obj1, Entry<Integer, Double> obj2) {
                return obj1.getValue().compareTo(obj2.getValue());
            }
        });

        //System.out.println("Sorted ArrayList: " + myList);
        //System.out.println("Original Map: " + map);

        // Set the global max value and local
        SplitData global = new SplitData(Double.MAX_VALUE);
        double current = Double.MAX_VALUE;

        // Initialize left set and right set as whole of dataset, and capture first value of Sorted List.
        HashSet<Integer> left = new HashSet<>();
        HashSet<Integer> right = new HashSet<>(dataset);
        double val = myList.get(0).getValue();

        // Iterate over the Sorted ArrayList
        for(int itr=0; itr < myList.size(); itr++) {
            left.add(myList.get(itr).getKey());

            // When the value matches to previous skips
            if (val == myList.get(itr).getValue()) {
                continue;
            }

            // Filter all data observation encountered from right till now. And get new value
            val = myList.get(itr).getValue();
            right.removeAll(left);

            // Compute Entropy or weighted average as per problem
            if (regression) {

                double leftavg = average(left, labelmap);
                double rightavg = average(right, labelmap);

                current = error(left, labelmap, leftavg) + error(right, labelmap, rightavg);

            } else {
                current = entropyWeighted(left, labelmap) + entropyWeighted(right, labelmap);
            }

            //System.out.println("Entropy " + current);
            //System.out.println("left " + left.size());
            //System.out.println("right " + right.size());

            if (left.size() == 0){
                current = Double.MAX_VALUE;
            }
            // Update previous if new error is smaller then observed till now
            if (current < global.error) {
                global.error = current;
                global.left = new HashSet<>(left);
                global.right = new HashSet<>(right);
                global.threshold = val;

            }
        }

        return global;
    }


    protected double average(HashSet<Integer> set, HashMap<Integer, Double> map) {
        double sum = 0;
        for (int key : set) {
            sum += map.get(key);
        }
        return sum / set.size();
    }


    protected double error(HashSet<Integer> set,  HashMap<Integer, Double> map, double avg) {
        double sum = 0;
        for(int key: set) {
            sum += Math.pow((map.get(key) - avg), 2);
        }
        return sum;
    }



    public double entropyWeighted(HashSet<Integer> set,  HashMap<Integer, Double> map) {
        int one = 0;
        for (int key : set) {
            if (map.get(key) == (double) 1){
                one++;
            }
        }

        int zero = set.size() - one;

        if (one == 0 || zero == 0) {
            return 0.0;
        }

        double probone = (double) one / set.size();
        double probzero = (double) zero / set.size();

        double val = probone * (Math.log(probone) / Math.log(2)) + probzero * (Math.log(probzero) / Math.log(2));

        double finalval = -val * ((double) set.size() / dataset.size());

        //System.out.println(finalval);
        return finalval;
    }


    public double majority(HashSet<Integer> set,  HashMap<Integer, Double> map){
        int ones = 0;
        for (int key : set) {
            if (map.get(key) == (double) 1){
                ones++;
            }
        }

        int zeros = set.size() - ones;

        if (ones >= zeros){
            return 1.0;
        } else {
            return  0.0;
        }
    }

}

