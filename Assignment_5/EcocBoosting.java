package neu.ml.assignment5;



import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

/**
 * Created by nikhilk on 11/20/15.
 */

    class EcocFunction{
        HashSet<Integer> ones;
        HashSet<Integer> zeros;
        double [] labels;

        public EcocFunction(int length){
            ones = new HashSet<>();
            zeros = new HashSet<>();
            labels = new double[length];
        }
    }

    class Hypoth{
        ArrayList<MDecisionStump.Stump> stlist = new ArrayList<>();
        ArrayList<Double> alphalist = new ArrayList<>();

        public Hypoth(){
            stlist = new ArrayList<>();
            alphalist = new ArrayList<>();

        }
    }


    public class EcocBoosting {
        double [] trainlabels;
        public double [][] train;

        double [] testlabels;
        public double [][] test;

        public double [][] table;

        public ArrayList<EcocFunction> functions;
        public ArrayList<Hypoth> hypothesis;

        public int runs;


        public void setData(double [][] train, double [] trainlabels, double [][] test , double [] testlabels, int runs){
            this.train = train;
            this.trainlabels = trainlabels;
            this.test = test;
            this.testlabels = testlabels;
            this.runs = runs;
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


        public ArrayList<EcocFunction> genLabelFunctions(double [][] data, double [] oldlabels){
            functions  = new ArrayList<>();


            for (int col = 0; col < data[0].length ; col ++){
                EcocFunction ef = new EcocFunction(oldlabels.length);

                // Iterate over row and put rownum in zeros and ones based on value
                for (int row =0 ; row < data.length ; row++){
                    if (data[row][col] == 0){
                        ef.zeros.add(row);
                    } else {
                        ef.ones.add(row);
                    }
                }

                // Generate the labelmap for current function
                for (int row =0; row < oldlabels.length; row++){
                    double val = oldlabels[row];

                    if (ef.zeros.contains((int) val)){
                        ef.labels[row] = 0;
                    } else {
                        ef.labels[row] = 1;
                    }
                }
                //System.out.println(ef.labelmap);
                // Add the object to list
                functions.add(ef);
            }

            return functions;
        }


        public void runBoosting() {
            hypothesis = new ArrayList<>();

            // Create and instance to generate Thresholds and RVFPairs
            MDecisionStump ds = new MDecisionStump();
            ds.genmaps(train);
            ArrayList<HashSet<Double>> thresholds = ds.thresholds;
            ArrayList<ArrayList<MDecisionStump.RFVPair>> rfvdata = ds.rfvdata;



            for (int col = 0; col < functions.size(); col++) {
                Hypoth hp = new Hypoth();
                double[] labels = functions.get(col).labels;

                // Create a object with new labels. Allocate precomputed Thresholds and RFVPairs of Columns again.
                ds = new MDecisionStump(labels, train);
                ds.thresholds = thresholds;
                ds.rfvdata = rfvdata;

                MDecisionStump.Stump best;
                double alpha;

                int counter = 1;

                while (counter <= runs) {
                    //Random rand = new Random();
                    //best = ds.getRandomStump(rand);
                    best = ds.getBestStump();
                    alpha = ds.computeAlpha(best);
                    hp.stlist.add(best);
                    hp.alphalist.add(alpha);

                    ds.updateDistribution(ds.distribution, alpha, best, train);

                    counter++;
                    //System.out.println(counter);
                    System.out.println(best.rounderr + " " +  best.feature);
                }
                if (hp.stlist.size() != 0) {
                    System.out.println(hp.stlist.get(hp.stlist.size() - 1).rounderr);
                }
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


        public double getPrediction(ArrayList<MDecisionStump.Stump> stlist,
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


        public void accCalc(double [] labels, double [] predictions){
            int match = 0;
            for (int itr = 0; itr < predictions.length ; itr++){
                if (predictions[itr] == labels[itr]){
                    match++;
                }
                System.out.println("Actual " + labels[itr] + " Prediction " + predictions[itr]);
            }
            double acc = (double) match / labels.length;
            System.out.print("Accuracy -> " + acc);
        }


        public static void main(String [] args){
            EcocBoosting eb = new EcocBoosting();


            // Populate data, generate Ecoc table and generate labels per function
            eb.table =  eb.genEcocTable(8, 20);
            eb.genLabelFunctions(eb.table, eb.trainlabels);

            //Perform boosting, make predictions,
            eb.runBoosting();
            double [] predictions = eb.makePredictions();

            eb.accCalc(eb.testlabels, predictions);


            /**
             be.table = be.genEcocTable();
             double [] my = {0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1};
             System.out.println(be.maxMatch(my));
             **/

        }
    }


