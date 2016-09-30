package neu.ml.assignment5;

import java.util.*;

/**
 * Created by nikhilk on 11/18/15.
 */



public class MDecisionStump {

    double [] distribution;
    double Zi;
    double [] labels;
    ArrayList<HashSet<Double>> thresholds;
    ArrayList<ArrayList<RFVPair>> rfvdata;
    double [][] mat;
    boolean random = false;


    public MDecisionStump(double [] labels, double [][] mat){
        this.labels = labels;
        this.mat = mat;
        AllocateZi();

    }


    public MDecisionStump(int size){
        labels = new double[size];
        AllocateZi();

    }

    public MDecisionStump(){
        // Empty constructor to initialize and use methods
    }


    public void setData(double [][] data){
        this.mat = data;
    }


    public void AllocateZi(){
        // Calculate the initial distribution
        Zi = labels.length;
        double constant = (double) 1 / labels.length;
        distribution = new double[labels.length];

        for(int itr = 0; itr < distribution.length; itr++){
            distribution[itr] = constant;
        }
    }

    public class Stump {
        double rounderr;
        double moderr;
        public int feature;
        public double threshold;

        public Stump(double val){
            moderr = val;
        }
    }


    public class RFVPair implements Comparable<RFVPair>{
        int row;
        double val;

        public RFVPair(int row, double val){
            this.row = row;
            this.val = val;
        }

        @Override
        public int compareTo(RFVPair rfv){
            Double val1 = this.val;
            Double val2 = rfv.val;
            return val1.compareTo(val2);
        }
    }

    public class RFVErr{
        int row;
        double current;
        double error;

        public RFVErr(int row, double cur, double err){
            this.row = row;
            current = cur;
            error = err;
        }
    }



    public void genmaps(double [][] data){
        thresholds = new ArrayList<>();
        rfvdata = new ArrayList<>();

        for (int col = 0 ; col < data[0].length ; col++){
            ArrayList<RFVPair> fvlist = new ArrayList<>();
            TreeSet<Double> fvals = new TreeSet<>();

            for (int row =0 ; row < data.length; row ++){
                fvlist.add(new RFVPair(row, data[row][col]));
                fvals.add(data[row][col]);
            }

            double min = fvals.first() - 1;
            double max = fvals.last() + 1;

            Iterator<Double> itr =  fvals.iterator();
            HashSet<Double> finval = new HashSet<>();

            double prev = itr.next();
            double curr;
            while(itr.hasNext()){
                curr = itr.next();
                finval.add( (prev + curr) / 2 );
                prev = curr;
            }

            finval.add(min);
            finval.add(max);
            Collections.sort(fvlist);

            thresholds.add(finval);
            rfvdata.add(fvlist);
        }
    }

    public double computeError(int col, double val){
        double error = 0;
        for (int row = 0; row < mat.length; row++) {
            if (mat[row][col] >= val) {
                if (labels[row] == 0){ error = error + distribution[row];}
            } else {
                if (labels[row] == 1){ error = error + distribution[row];}
            }
        }
        return  error;
    }


    public RFVErr computeError(double val, RFVErr rfve, ArrayList<RFVPair> list){
        double error = rfve.current;
        int start =rfve.row + 1;

        for (int row =start ; row < list.size(); row++) {
            RFVPair pair = list.get(row);
            if (pair.val <= val ) {
                if (labels[pair.row] == 1) { error = error + distribution[pair.row];}
                rfve.current = error;
                rfve.row = row;
            } else {
                if (labels[pair.row] == 0){ error = error + distribution[pair.row];}
            }
        }
        rfve.error = error;
        return  rfve;
    }


    public Stump bestLocalStump(HashSet<Double> set, ArrayList<RFVPair> list) {
        Stump current = new Stump(Double.MIN_VALUE);
        RFVErr rfve = new RFVErr(0,0, Double.MIN_VALUE);

        for (double val : set){
            rfve = computeError(val, rfve, list);

            if (Math.abs(.5 - rfve.error) > current.moderr) {
                current.moderr = Math.abs(.5 - rfve.error);
                current.rounderr = rfve.error;
                current.threshold = val;
            }
        }
        //System.out.println("Error " + current.error);
        return current;
    }


    public Stump getBestStump(){
        Stump best = new Stump(Double.MIN_VALUE);
        Stump current;
        for(int col = 0; col<thresholds.size(); col++){
            // Now we have map of dataset to feature values. computeCost now.
            current = bestLocalStump(thresholds.get(col), rfvdata.get(col));

            //System.out.println("Stump " + current.error);

            if (current.moderr > best.moderr) {
                best = current;
                best.feature = col;
            }
        }
        //System.out.println("Feature " + best.feature);
        return  best;
    }


    public Stump getRandomStump(Random rand){
        int fid = rand.nextInt((thresholds.size()));

        HashSet<Double> set = thresholds.get(fid);
        int index = rand.nextInt(set.size());

        ArrayList<Double> arl = new ArrayList<>(set);
        double threshold = arl.get(index);

        double error = computeError(fid, threshold);
        Stump best = new Stump(error);
        best.feature = fid;
        best.threshold = threshold;
        best.rounderr = error;

        return best;
    }


    public double computeAlpha(Stump st){
        double val = .5 * Math.log((1 - st.rounderr) / st.rounderr);
        return val;
    }

    public void updateDistribution(double [] dstr, double alpha, Stump st, double[][] mat){
        double znew = 0;
        int feature = st.feature;
        double threshold = st.threshold;

        for (int itr = 0; itr < dstr.length; itr ++){
            int htx = 1;
            if (mat[itr][feature] >= threshold){
                if (labels[itr] == 1){
                    htx = -1;
                }
            } else {
                if (labels[itr] == 0){
                    htx = -1;
                }
            }

            dstr[itr] = dstr[itr] * Math.exp(alpha * htx);
            znew = znew + dstr[itr];
        }

        for (int itr = 0; itr < dstr.length; itr ++){
            dstr[itr] = dstr[itr] / znew;
        }

        //System.out.println("Zi " + znew + " Threshold " + threshold + " Feature " + feature);
    }

}



