package neu.ml.assignment4;

/**
 * Created by nikhilk on 10/28/15.
 */

import neu.ml.assignment1.*;

import java.util.HashSet;

public class AdaBoosting {


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
            test.add(key);
            train.remove(key);


            PredictClassification prc = new PredictClassification();
            prc.buildMatrix(ctd.datamap, ctd.bucketmap, train, true);
            prc.buildMatrix(ctd.datamap, ctd.bucketmap, test, false);

            IterationStats its = new IterationStats(prc.trainmat, prc.trainmap, prc.testmat, prc.testmap);
            its.tillConvergence();
            System.exit(0);
        }
    }


    public static void main(String [] args) {
        CrossTrainData ctd = new CrossTrainData();
        ctd.loadFile();

        AdaBoosting ada = new AdaBoosting();
        ada.buildAndPredict(ctd);

    }
}
