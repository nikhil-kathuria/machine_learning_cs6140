package neu.ml.assignment5;

/**
 * Created by nikhilk on 11/11/15.
 */


import neu.ml.assignment4.IterationStats;
import org.ejml.simple.SimpleMatrix;

import java.util.*;

import neu.ml.assignment4.DecisionStump;
import neu.ml.assignment4.DecisionStump.Stump;


public class BestFeatures {
    double [][] data;
    double [] labels;

    ArrayList<Double> alphalist;
    ArrayList<Stump> stumplist;
    int rows;
    int columns;

    public void populateData(HashMap<Integer, String> map, String splitstring) {
        rows = map.size();
        columns = map.get(0).split(splitstring).length;
        double [] label = new double [rows];

        double[][] data2d = new double[rows][columns - 1];
        for (int row = 0 ; row < rows ; row++) {

            // Split the data
            String[] datarow = map.get(row).split(splitstring);
            for(int col = 0; col < columns - 1; col++) {
                data2d[row][col] = Double.valueOf(datarow[col]);
            }
            label[row]=  Double.valueOf(datarow[columns - 1]);
        }
        data = data2d;
        labels = label;
    }


    public double rowmargin(int row){
        double margin = 0;
        double hypoth;
        int Yx;

        for (int col=0; col < alphalist.size(); col++){
            int feature = stumplist.get(col).feature;
            double threshold = stumplist.get(col).threshold;

            if (data[row][feature] >= threshold) {
                hypoth = alphalist.get(col);
            } else {
                hypoth = -alphalist.get(col);
            }

            if (labels[row] == 0){
                Yx = -1;
            } else {
                Yx = 1;
            }
            margin = margin + Yx * hypoth;

        }
        return margin;
    }


    public double marginData(){
        double summargin = 0;
        double [] marginrow = new double[data.length];

        for (int  row =0; row < data.length ; row++){
            marginrow[row] = rowmargin(row);
            summargin += marginrow[row];
        }
        return summargin;
    }


    public double fmarginData(int fid) {
        double margin = 0;
        double hypoth;
        int Yx;

        for (int row = 0; row < rows; row++) {

            for (int itr = 0; itr < stumplist.size(); itr++) {

                if (stumplist.get(itr).feature == fid) {
                    if (data[row][fid] >= stumplist.get(itr).threshold) {
                        hypoth = alphalist.get(itr);
                    } else {
                        hypoth = -alphalist.get(itr);
                    }

                    if (labels[row] == 0) {
                        Yx = -1;
                    } else {
                        Yx = 1;
                    }

                    margin = + margin + Yx * hypoth;
                }
            }
        }
        return margin;
    }



    public HashMap<Integer, Double> featureMargin(double summargin){
        double fmargin;
        HashMap<Integer, Double> fmap = new HashMap<>();

        for (int col =0 ; col < columns; col++){
             fmargin = fmarginData(col);
            System.out.println("Feature " + col);
            fmap.put(col, fmargin/ summargin);
        }
        return fmap;
    }


    public void printFeatures(HashMap<Integer, Double> fmap){
        ArrayList<Map.Entry<Integer, Double>> myList = new ArrayList<>(fmap.entrySet());
        Collections.sort(myList, new Comparator<Map.Entry<Integer, Double>>() {
            @Override
            public int compare(Map.Entry<Integer, Double> obj1, Map.Entry<Integer, Double> obj2) {
                return obj2.getValue().compareTo(obj1.getValue());
            }
        });

        for (int row = 0; row < myList.size(); row++ ){
            System.out.println("Feature ID " + myList.get(row).getKey() + " Feature Fraction Margin " + myList.get(row).getValue());
        }

    }


    public HashMap<Integer, Double> map(double [] label){
        HashMap<Integer, Double> mymap = new HashMap<>();

        for (int row=0; row < label.length; row++){
            mymap.put(row, label[row]);
        }

        return mymap;
    }

    public static void  main(String [] args){
        ParseData pd = new ParseData();
        pd.splitstring = ",";
        BestFeatures bf = new BestFeatures();

        bf.populateData(pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt"), pd.splitstring);

        HashMap<Integer, Double> mymap = bf.map(bf.labels);

        IterationStats its = new IterationStats(new SimpleMatrix(bf.data), mymap,new SimpleMatrix(bf.data), mymap);
        its.runs = 300;
        its.tillConvergence();
        System.out.println(its.averageError(new SimpleMatrix(bf.data), mymap));



        bf.stumplist = its.stlist; bf.alphalist = its.alphalist;
        double summargin = bf.marginData();

        System.out.println(summargin);
        HashMap<Integer, Double> fmap = bf.featureMargin(summargin);
        System.out.println(summargin);

        bf.printFeatures(fmap);

    }

}
