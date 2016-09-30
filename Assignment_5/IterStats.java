package neu.ml.assignment5;


import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by nikhilk on 11/20/15.
 */

public class IterStats {
    public double[][] train;
    public double[] trainlabels;

    public double[][] test;
    public double[] testlabels;

    public ArrayList<MDecisionStump.Stump> stlist;
    public ArrayList<Double> alphalist;

    public int runs;

    public IterStats(double [][] train, double [] trainlabels, double [][] test, double [] testlabels){
        this.train = train;
        this.trainlabels = trainlabels;
        this.test = test;
        this.testlabels = testlabels;

        alphalist = new ArrayList<>();
        stlist = new ArrayList<>();

        runs = 100;
    }


    public double averageError(double [][] mat, double [] label){
        double [] predictions = new double[mat.length];
        double [] predictvalues = new double[mat.length];

        for (int row = 0; row < mat.length; row++){
            double prediction = 0;

            for (int itr=0; itr < stlist.size(); itr++){
                int feature = stlist.get(itr).feature;
                double threshold = stlist.get(itr).threshold;

                if (mat[row][feature] >= threshold) {
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

        // For plotting AUC
        //this.predictions = predictions;
        //this.predictvalues = predictvalues ;

        // Return the average Error i.e.
        int error = 0;
        for (int itr = 0; itr < predictions.length; itr++){
            if(predictions[itr] != label[itr]){
                error++;
            }
        }
        return (double) error / predictions.length;
    }


    public void tillConvergence() {
        double prev = 0;
        double trainerr;
        double testerr;

        MDecisionStump ds = new MDecisionStump(trainlabels, train);
        ds.genmaps(train);
        MDecisionStump.Stump best;
        double alpha;
        double area;

        int counter = 1;


        while (counter <= runs) {
            best = ds.getBestStump();
            alpha = ds.computeAlpha(best);

            stlist.add(best);
            alphalist.add(alpha);

            ds.updateDistribution(ds.distribution, alpha, best, train);

            //System.out.println(ds.distribution[0]);


            trainerr = averageError(train, trainlabels);
            testerr = averageError(test, testlabels);


            System.out.println("Round " + counter + " Feature " + best.feature + " Threshold " + best.threshold + " Round_Err "
                    + best.rounderr + " Train Error " + trainerr + " Test Error " + testerr);

            counter++;

        }
    }


}