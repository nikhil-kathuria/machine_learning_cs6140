package neu.ml.assignment4;

import org.ejml.simple.SimpleMatrix;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.List;
import java.util.Collections;
import java.util.ArrayList;
import java.util.Comparator;


/**
 * Created by nikhilk on 10/28/15.
 */


public class Old_DecisionStump {
    double [] distribution;
    double Zi;
    HashMap<Integer, Double> labelmap;
    ArrayList<List<Entry<Integer, Double>>> holder;


    public Old_DecisionStump(HashMap<Integer, Double> labelmap, SimpleMatrix mat){
        this.labelmap = labelmap;
        holder = genmaps(mat);

        Zi = labelmap.size();
        double constant = (double) 1 / labelmap.size();
        distribution = new double[labelmap.size()];

        for(int itr =0; itr < distribution.length; itr++){
            distribution[itr] = constant;
        }

    }

    public class Stump {
        double error;
        int feature;
        double threshold;

        public Stump(double val){
            error = val;
        }
    }


    public List<Map.Entry<Integer, Double>> sortMyMap(HashMap<Integer, Double>map){

        // Sort the Map based on values
        List<Map.Entry<Integer, Double>> myList = new ArrayList<Map.Entry<Integer, Double>>(map.entrySet());
        Collections.sort(myList, new Comparator<Map.Entry<Integer, Double>>() {
            @Override
            public int compare(Entry<Integer, Double> obj1, Entry<Integer, Double> obj2) {
                return obj1.getValue().compareTo(obj2.getValue());
            }
        });

        return myList;

    }

    public ArrayList<List<Entry<Integer, Double>>> genmaps(SimpleMatrix mat){
        ArrayList<List<Entry<Integer, Double>>> holder = new ArrayList<>() ;

        for (int col = 0 ; col < mat.numCols(); col++){
            HashMap<Integer, Double> map = new HashMap<>();
            for (int row =0 ; row < mat.numRows(); row ++){
                map.put(row, mat.get(row, col));

            }
            holder.add(sortMyMap(map));
        }
        return  holder;
    }


    public double computeError(List<Entry<Integer, Double>> list, int pos){
        double error = 0;
        for (int row = 0; row < list.size(); row++) {
            if (pos >= row) {
                if (labelmap.get(row) == 0){ error = error + distribution[row];}
            } else {
                if (labelmap.get(row) == 1){ error = error + distribution[row];}
            }

        }
        return  error;
    }


    public Stump bestLocalStump(List<Entry<Integer, Double>> list) {
        Stump current = new Stump(Double.MIN_VALUE);
        double val = list.get(0).getValue();
        for (int row = 0; row < list.size(); row++) {
            list.get(row).getKey();


            // When the value matches to previous skips
            if (val == list.get(row).getValue()) {
                continue;
            }

            val = list.get(row).getValue();
            double error = computeError(list, row);

            if (Math.abs(.5 - error) > current.error) {
                System.out.println("Error " + error);
                current.error = Math.abs(.5 - error);
                current.threshold = list.get(row).getValue();
            }
        }
        return current;
    }


    public Stump getBestStump(){
        Stump best = new Stump(Double.MIN_VALUE);
        Stump current;
        for(int col = 0; col<holder.size(); col++){
            // Now we have map of dataset to feature values. computeCost now.
            current = bestLocalStump(holder.get(col));

            //System.out.println("Stump " + current.error);

            if (current.error > best.error) {
                best = current;
                best.feature = col;
            }
        }
        //System.out.println("Feature " + best.feature);
        return  best;
    }

    public double computeAlpha(Stump st){
        double val = .5 * Math.log((1 - st.error) / st.error);
        return val;
    }

    public void updateDistribution(double [] dstr, double alpha, Stump st, SimpleMatrix mat){
        double znew = 0;
        int feature = st.feature;
        double threshold = st.threshold;

        int htx = 0;
        for (int itr = 0; itr < dstr.length; itr ++){
            if (mat.get(itr, feature) >= threshold){
                htx = 1;
            } else { htx = -1; }

            dstr[itr] = (dstr[itr] / Zi) * Math.exp(alpha * htx);
            znew = znew + dstr[itr];

        }
        this.Zi = znew;
        System.out.println("Zi " + Zi);
    }

}
