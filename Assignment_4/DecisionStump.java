package neu.ml.assignment4;

import org.ejml.simple.SimpleMatrix;

import java.util.*;
import java.util.Random;

/**
 * Created by nikhilk on 10/28/15.
 */


public class DecisionStump {
    double [] distribution;
    double Zi;
    HashMap<Integer, Double> labelmap;
    ArrayList<HashSet<Double>> holder;
    SimpleMatrix mat;
    boolean random = false;


    public DecisionStump(HashMap<Integer, Double> labelmap, SimpleMatrix mat){
        this.labelmap = labelmap;
        this.mat = mat;
        holder = genmaps(mat);

        /**double size = 0;
        for (HashSet<Double> ds : holder){
            System.out.println(ds.size());
            size += ds.size();
        }
        System.out.println("Final " + size);
        System.exit(0); **/

        Zi = labelmap.size();
        double constant = (double) 1 / labelmap.size();
        distribution = new double[labelmap.size()];

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


    public ArrayList<HashSet<Double>> genmaps(SimpleMatrix mat){
        holder = new ArrayList<>();

        for (int col = 0 ; col < mat.numCols(); col++){
            TreeSet<Double> set = new TreeSet<>();
            HashSet<Double> thresholds = new HashSet<>();

            for (int row =0 ; row < mat.numRows(); row ++){
                set.add(mat.get(row, col));
            }

            double min = set.first() - 1;
            double max = set.last() + 1;

            Iterator<Double> itr =  set.iterator();

            double prev = itr.next();
            double curr;
            while(itr.hasNext()){
                curr = itr.next();
                thresholds.add( (prev + curr) / 2 );
                prev = curr;
            }

            thresholds.add(min);
            thresholds.add(max);

            holder.add(thresholds);
        }
        return  holder;
    }


    public double computeError(int col, double val){
        double error = 0;
        for (int row = 0; row < mat.numRows(); row++) {
            if (mat.get(row, col) >= val) {
                if (labelmap.get(row) == 0){ error = error + distribution[row];}
            } else {
                if (labelmap.get(row) == 1){ error = error + distribution[row];}
            }
        }
        return  error;
    }


    public Stump bestLocalStump(int col, HashSet<Double> set) {
        Stump current = new Stump(Double.MIN_VALUE);
        for (double val : set){
            double error = computeError(col, val);

            if (Math.abs(.5 - error) > current.moderr) {
                current.moderr = Math.abs(.5 - error);
                current.rounderr = error;
                current.threshold = val;
            }
        }
        //System.out.println("Error " + current.error);
        return current;
    }


    public Stump getBestStump(){
        Stump best = new Stump(Double.MIN_VALUE);
        Stump current;
        for(int col = 0; col<holder.size(); col++){
            // Now we have map of dataset to feature values. computeCost now.
            current = bestLocalStump(col, holder.get(col));

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
        int fid = rand.nextInt((holder.size()));

        HashSet<Double> set = holder.get(fid);
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

    public void updateDistribution(double [] dstr, double alpha, Stump st, SimpleMatrix mat){
        double znew = 0;
        int feature = st.feature;
        double threshold = st.threshold;

        for (int itr = 0; itr < dstr.length; itr ++){
            int htx = 1;
            if (mat.get(itr, feature) >= threshold){
                if (labelmap.get(itr) == 1){
                    htx = -1;
                }
            } else {
                if (labelmap.get(itr) == 0){
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
