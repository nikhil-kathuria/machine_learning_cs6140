package neu.ml.assignment6;

/**
 * Created by nikhilk on 12/12/15.
 */


import java.util.HashSet;

public class SMOSpam {
    double [][] train;
    double [] trainlabels;

    double [][] test;
    double [] testlabels;

    public double [] smoLabels(double [] labels){
        for (int itr=0; itr < labels.length ; itr++){
            if (labels[itr] == 0){
                labels[itr] = -1;
            }
        }
        return labels;
    }

    public void runSMO(ParseSpam pm) {
        // Total buckets and tree holder
        HashSet<Integer> total = new HashSet<>(pm.bucketmap.keySet());
        double summse = 0;
        double sumacc = 0;
        double acc = 0;
        double mse = 0;

        Normalize nm = new Normalize();

        for (int key : total) {
            // Assign test and train. Update train
            HashSet<Integer> trainset = new HashSet<>(total);
            HashSet<Integer> testset = new HashSet<>(key);
            testset.add(key);
            trainset.remove(key);

            pm.buildData(pm.datamap, pm.bucketmap, testset);
            test = nm.normalizeall(pm.data);
            testlabels = pm.labels;


            pm.buildData(pm.datamap, pm.bucketmap, trainset);
            train =  nm.normalizeall(pm.data);
            trainlabels = smoLabels(pm.labels);

            SMOSolver sms = new SMOSolver(train, trainlabels);
            sms.Solver();
            sms.accCalc(test, testlabels);

            //SMOSimplified smp = new SMOSimplified(train, trainlabels);
            //smp.runSMO();
            //smp.accCalc(testlabels);

            //System.exit(0);
        }
    }

    public static void main(String [] args){
        ParseSpam pm = new ParseSpam();
        pm.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt");

        SMOSpam smp = new SMOSpam();
        smp.runSMO(pm);

    }
}
