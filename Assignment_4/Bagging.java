package neu.ml.assignment4;

import neu.ml.assignment1.DecisionTree;
import neu.ml.assignment1.PredictClassification;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;



// import java.util;

/**
 * Created by nikhilk on 10/30/15.
 */

public class Bagging {
    SimpleMatrix data;
    HashMap<Integer, Double> map;


    public void bagAndTrain(){
        ParseSpam pm = new ParseSpam();
        // Added now and removed from constructor of ParseSpam.
        pm.genTrain(new HashSet<Integer>(pm.labelmap.keySet()));
        PredictClassification pc = new PredictClassification();


        SimpleMatrix mat = new SimpleMatrix(pm.data);
        double [] finalprediction = new double[pm.data.length];

        for (ArrayList<Integer> list : pm.trainlists) {
            setDataAndMap(list, pm);
            DecisionTree head = new DecisionTree(new HashSet<>(map.keySet()), map, data, false);
            head.buildTree();


            double[] prediction = pc.predictClass(head, mat);
            double acc = pc.acccalc(prediction, pm.labelmap);
            System.out.println("Accuracy " +  acc);
            updatePrediction(finalprediction, prediction);


        }
        finalprediction = finalPredict(finalprediction, pm.trainlists.size());
        double acc = pc.acccalc(finalprediction, pm.labelmap);
        System.out.println("Final Accuracy " +  acc);
    }


    public void updatePrediction(double []global, double [] local){
        for (int itr =0 ; itr < global.length; itr++){
            global[itr] = global[itr] + local[itr];
        }
    }


    public double[] finalPredict(double [] global, int size){
        double [] prediction = new double [global.length];
        for (int itr = 0; itr < global.length; itr++){
            double val = global[itr] / size;
            if (val > .5) {
                prediction[itr] = 1;
            } else {
                prediction[itr] = 0;
            }
        }

        return prediction;
    }


    public void setDataAndMap(ArrayList<Integer> list, ParseSpam pm){
        double [][] newdata = new double[list.size()][pm.data[0].length];

        /**
        System.out.println(list.size());
        System.out.println(pm.data.length);
        System.out.println(pm.data[0].length);
        System.out.println(newdata.length);
        System.out.println(newdata[0].length);
        **/

        HashMap<Integer, Double> newmap = new HashMap<>();

        for (int row = 0 ; row < newdata.length; row++){
            for (int col =0 ; col < newdata[row].length; col++){
                newdata[row][col] = pm.data[list.get(row)][col];
            }
            newmap.put(row, pm.labelmap.get(list.get(row)));
        }
        data = new SimpleMatrix(newdata);
        map = newmap;



    }


    public static void main(String [] args){
        Bagging bag = new Bagging();
        bag.bagAndTrain();

    }


}
