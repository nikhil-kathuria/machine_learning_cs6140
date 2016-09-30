package neu.ml.assignment4;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by nikhilk on 11/4/15.


 */

public class ParseEcoc {
    int rows;
    int columns = 1754;
    public String splitstring = "\\s+";
    public String wordsplit = ":";
    int numclass = 8;
    HashMap<Integer, Double> labelmap;
    double [][] data;


    public HashMap<Integer, String> readFile(String fileName) {
        HashMap<Integer, String> data = new HashMap<Integer, String>();

        try{
            //Create object of FileReader
            FileReader inputFile = new FileReader(fileName);

            //Instantiate the BufferedReader Class
            BufferedReader bufferReader = new BufferedReader(inputFile);

            //Variable to hold the one line data and line count
            String line; int rownum = 0;

            // Read file line by line and print on the console
            while ((line = bufferReader.readLine()) != null)   {
                data.put(rownum, line.trim());
                rownum ++;

            }
            //Close the buffer reader and FileReader
            bufferReader.close();
            inputFile.close();

            if (! data.isEmpty()) {
                rows = data.size();
                //columns = data.get(0).split(splitstring).length;
            }

        } catch(Exception e) {
            e.printStackTrace(System.out);
            //System.out.println("Error while reading file line by line: " + e.getMessage());
        }

        return data;
    }


    public void buildMat(HashMap<Integer, String> map){
        String [] line;
        labelmap = new HashMap<>();
        data = new double[rows][columns];
        //System.out.println(map.get(0));

        for(int row = 0 ; row < map.size() ;row++ ){
            String str = map.get(row);
            line = str.split(splitstring);
            String [] word;

            for (int col =0 ; col < line.length; col++){
                if(col == 0){
                    labelmap.put(row, Double.valueOf(line[col]));
                } else {
                    word = line[col].split(wordsplit);
                    //System.out.println(word.length);
                    data[row][Integer.valueOf(word[0])] = Double.valueOf(word[1]);
                }
            }
        }

    }


    public static void main(String [] args){
        ParseEcoc ec = new ParseEcoc();
        HashMap<Integer, String> map = ec.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_4/ECOC/8newsgroup/test.trec/feature_matrix.txt");
        ec.buildMat(map);

    }
}
