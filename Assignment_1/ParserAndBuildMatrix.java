package neu.ml.assignment1;

/**
 * Created by nikhilk on 9/12/15.
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
//import java.util.HashSet;
import org.ejml.simple.SimpleMatrix;
//import org.ejml.data.DenseMatrix64F;

/**
 * This example code shows you how to read file in Java
 *
 *
 */

public class ParserAndBuildMatrix {

    public SimpleMatrix matrix;
    public HashMap<Integer, Double> labelmap;
    int rows;
    int columns;

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
                //System.out.println(line);
            }
            //Close the buffer reader and FileReader
            bufferReader.close();
            inputFile.close();

            if (! data.isEmpty()) {
                rows = data.size();
                //columns = data.get(0).split("\\s+").length;
                columns = data.get(0).split("\\s+").length;
            }

        } catch(Exception e) {
            e.printStackTrace(System.out);
            //System.out.println("Error while reading file line by line: " + e.getMessage());
        }

        return data;
    }

    public void buildMatrix(HashMap<Integer, String> map) {
        HashMap<Integer, Double> labelmap = new HashMap<Integer, Double>();

        double[][] data2d = new double[rows][columns - 1];
        for (int row = 0 ; row < rows ; row++) {
            // Change for hosing data and spam data
            //String[] datarow = map.get(row).split("\\s+");
            String[] datarow = map.get(row).split("\\s+");
            for(int col = 0; col < columns - 1; col++) {
                data2d[row][col] = Double.valueOf(datarow[col]);
            }
            labelmap.put(row, Double.valueOf(datarow[columns - 1]));
        }

        matrix = new SimpleMatrix(data2d);
        this.labelmap =labelmap;
        //System.out.println(matrix.get(0, 0));
        }


    public static void main(String[] args) {
        ParserAndBuildMatrix par = new ParserAndBuildMatrix();
        HashMap<Integer, String> data =
                par.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/housing_train.txt");
        par.buildMatrix(data);
        //System.out.println(par.matrix);
    }
}
