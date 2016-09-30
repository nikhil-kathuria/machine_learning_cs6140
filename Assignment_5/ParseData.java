package neu.ml.assignment5;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;

/**
 * Created by nikhilk on 11/10/15.
 */

public class ParseData {
    public String splitstring = "\\s+";
    int rows;
    int columns;

    public HashMap<Integer, String> readFile(String fileName) {

        HashMap<Integer, String> data = new HashMap<>();

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
                //System.out.println(line);
            }
            //Close the buffer reader and FileReader
            bufferReader.close();
            inputFile.close();

            if (! data.isEmpty()) {
                rows = data.size();
                //columns = data.get(0).split("\\s+").length;
                columns = data.get(0).split(splitstring).length;
            }

        } catch(Exception e) {
            e.printStackTrace(System.out);
            //System.out.println("Error while reading file line by line: " + e.getMessage());
        }

        return data;
    }


    public double [] populateLabels(String filename){
        HashMap<Integer, String> mymap = readFile(filename);
        double [] labels = new double[mymap.size()];

        for (int itr=0; itr< mymap.size(); itr++){
            labels[itr] = Double.valueOf(mymap.get(itr));
        }

        return  labels;
    }

    public double[][] populateData(HashMap<Integer, String> mymap){
        double [][] data = new double[rows][columns];

        for (int row = 0; row < rows; row++){
            String [] datarow = mymap.get(row).split(splitstring);

            for (int col=0; col < datarow.length; col++){
                data[row][col] = Double.valueOf(datarow[col]);
            }
        }
        return data;
    }


    public static void main(String [] args){
        ParseData pd = new ParseData();
        pd.populateData(pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/train_feature.txt"));

        pd.populateLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/spam_polluted/test_label.txt");

    }
}

