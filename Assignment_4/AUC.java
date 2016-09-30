package neu.ml.assignment4;

import java.util.*;

/**
 * Created by nikhilk on 11/6/15.
 */


class Struct implements Comparable<Struct>{
    double prediction;
    double value;
    double actual;

    public Struct(double predict, double val, double act){
        prediction = predict;
        value = val;
        actual = act;

    }

    @Override
    public int compareTo(Struct st){
        Double val1 = this.value;
        Double val2 = st.value;
        return val1.compareTo(val2);
    }


}

public class AUC {
    HashMap<Integer, Double> map;
    double [] prediction;
    double [] value;

    ArrayList<Double> tprate;
    ArrayList<Double> fprate;

    public AUC(double [] prediction, double [] predicvalues, HashMap<Integer, Double> map){
        this.map = map;
        this.value = predicvalues;
        this.prediction = prediction;
    }

    public void TpFpRate(){
        ArrayList<Struct> myList = new ArrayList<>();

        tprate = new ArrayList<>();
        fprate = new ArrayList<>();

        for (int row= 0; row < prediction.length; row ++ ){
            Struct st = new Struct(prediction[row], value[row] ,map.get(row));
            myList.add(st);
        }

        Collections.sort(myList);
        for (int row=0; row < myList.size(); row++ ){
            int tp = 0;
            int fp = 0;
            int tn = 0;
            int fn = 0;

            for (int cur=0 ;cur < myList.size(); cur++){
                if (cur <= row){
                    if (myList.get(cur).prediction == myList.get(cur).actual ){
                        tp++;
                    } else {
                        fp++;
                    }
                } else {
                    if (myList.get(cur).prediction == myList.get(cur).actual ){
                        tn++;
                    } else {
                        fn++;
                    }
                }
            }
            tprate.add((double) tp / (tp + fn));
            fprate.add((double) fp / (fp + tn));

        }

    }

    public double aucCalc(){
        // fprate is X axis
        // tprate is Y axis

        double area=0;
        for (int itr = 1; itr <fprate.size(); itr ++){
            area += (fprate.get(itr) - fprate.get(itr -1)) * (tprate.get(itr) + tprate.get(itr - 1));
        }
        return .5 * area;
    }
}
