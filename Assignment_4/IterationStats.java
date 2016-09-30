package neu.ml.assignment4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import neu.ml.assignment1.CrossTrainData;
import neu.ml.assignment1.DecisionTree;
import neu.ml.assignment1.PredictClassification;
import neu.ml.assignment4.DecisionStump.*;
import org.ejml.simple.SimpleMatrix;

/**
 * Created by nikhilk on 10/29/15.
 */

public class IterationStats {
    public SimpleMatrix testmat;
    public SimpleMatrix trainmat;

    public HashMap<Integer, Double> testmap;
    public HashMap<Integer, Double> trainmap;

    public ArrayList<Stump> stlist = new ArrayList<>();
    public ArrayList<Double> alphalist = new ArrayList<>();

    public int runs = 100;
    double [] predictions;
    double [] predictvalues;

    // Constructor
    public IterationStats(SimpleMatrix trainmat, HashMap<Integer, Double> trainmap,
                          SimpleMatrix testmat, HashMap<Integer, Double> testmap){
        this.testmat = testmat;
        this.trainmat = trainmat;
        this.testmap = testmap;
        this.trainmap = trainmap;
    }


    public double averageError(SimpleMatrix mat, HashMap<Integer, Double> map){
        double [] predictions = new double[mat.numRows()];
        double [] predictvalues = new double[mat.numRows()];

        for (int row = 0; row < mat.numRows(); row++){
            double prediction = 0;

            for (int itr=0; itr < stlist.size(); itr++){
                int feature = stlist.get(itr).feature;
                double threshold = stlist.get(itr).threshold;

                if (mat.get(row, feature) >= threshold) {
                    prediction = prediction + alphalist.get(itr);
                } else {
                    prediction = prediction - alphalist.get(itr);
                }
            }
            if (prediction < 0) {
                predictions[row] = 0;
            }
            else {
                predictions[row] = 1;
            }
            // Store actual prediction value and not classification
            predictvalues[row] = prediction;
        }

        this.predictions = predictions;
        this.predictvalues = predictvalues ;

        // Return the average Error i.e.
        int error = 0;
        for (int itr = 0; itr < predictions.length; itr++){
            if(predictions[itr] != map.get(itr)){
                error++;
            }
        }
        return (double) error / predictions.length;
    }


    public void tillConvergence(){
        double prev = 0;
        double trainerr;
        double testerr;

        DecisionStump ds = new DecisionStump(trainmap, trainmat);
        Stump best;
        double alpha;
        double area;

        int counter = 1;


        while( counter <= runs ){

            Random rand = new Random();
            best = ds.getRandomStump(rand);

            //best = ds.getBestStump();
            alpha = ds.computeAlpha(best);
            stlist.add(best);
            alphalist.add(alpha);

            ds.updateDistribution(ds.distribution, alpha, best, trainmat);


            trainerr = averageError(trainmat, trainmap);
            testerr = averageError(testmat, testmap);

            /**
            AUC auc = new AUC(predictions, predictvalues, testmap);
            auc.TpFpRate();
            area = auc.aucCalc();

            System.out.println("Round " + counter + " Feature " + best.feature + " Round_Err "
                    + best.rounderr + " Train_Err " + tre + " Test_Err " + tee + " AUC " + area);
             **/

            counter++;

            System.out.println("Round " + counter + " Train Error " + trainerr + " Test Error " + testerr);
        }
        //tre = averageError(trainmat, trainmap);
        //tee = averageError(testmat, testmap);
        //System.out.println("Train Error " + tre + " Test Error " + tee);
    }
}


