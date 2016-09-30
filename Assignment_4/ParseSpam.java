package neu.ml.assignment4;

import neu.ml.assignment1.ParserAndBuildMatrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

/**
 * Created by nikhilk on 10/30/15.
 */
public class ParseSpam {
    int numRounds = 50;
    double[][] data;
    HashMap<Integer, Double> labelmap;
    String splitString = ",";
    ArrayList<ArrayList<Integer>> trainlists;
    // Other split pattern "\\s+"

    public ParseSpam(){
        HashMap<Integer, String> data = fetchData();
        buildMatrix(data);

    }

    public HashMap<Integer, String> fetchData() {
        ParserAndBuildMatrix pab = new ParserAndBuildMatrix();
        HashMap<Integer, String> data =
                pab.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt");
        return data;
    }

    public void buildMatrix(HashMap<Integer, String> map) {
        HashMap<Integer, Double> labelmap = new HashMap<Integer, Double>();
        int rows = map.size();
        int columns = columns = map.get(0).split(splitString).length;

        double[][] data2d = new double[rows][columns - 1];
        for (int row = 0 ; row < rows ; row++) {
            String[] datarow = map.get(row).split(splitString);
            for(int col = 0; col < columns - 1; col++) {
                data2d[row][col] = Double.valueOf(datarow[col]);
            }
            labelmap.put(row, Double.valueOf(datarow[columns - 1]));
        }

        data = data2d;
        this.labelmap =labelmap;

    }

    public void genTrain(HashSet<Integer> keys){
        ArrayList<ArrayList<Integer>> holder = new ArrayList<>();
        Random rand = new Random();
        int min = 0;
        int max =keys.size();
        for (int itr = 0 ; itr < numRounds; itr++ ){
            ArrayList<Integer> list = new ArrayList<>();

            while(list.size() != max){
                int randomNum = rand.nextInt(keys.size());
                //int randomNum = rand.nextInt((max - min) + 1) + min;
                list.add(randomNum);
            }
            holder.add(list);
        }
        trainlists = holder;
    }


}

