package neu.ml.assignment1;

import org.ejml.simple.SimpleMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Collections;


/**
 * Created by nikhilk on 9/16/15.
 */
public class CrossTrainData {
    //HashMap<Integer, Double> labelmap;
    int rows;
    int columns;
    public HashMap<Integer, HashSet<Integer>> bucketmap;
    public HashMap<Integer, String> datamap;

    public void readFile(String fileName) {
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
                columns = data.get(0).split("\\s+").length;
            }

        } catch(Exception e) {
            e.printStackTrace(System.out);
            //System.out.println("Error while reading file line by line: " + e.getMessage());
        }

        datamap = data;
        genBuckets(new HashSet<Integer>(data.keySet()));
    }

    public void genBuckets(HashSet<Integer> keys) {
        int counter = 1;
        int buckets = 10;
        int bucketsize = (keys.size() / buckets) + 1;

        ArrayList<Integer> mylist = new ArrayList<>(keys);

        Collections.shuffle(mylist);

        HashMap<Integer, HashSet<Integer>> map = new HashMap<>();
        HashSet<Integer> elements = new HashSet<>();

        for (int key : mylist){
            if (counter % bucketsize == 0){
                map.put(counter / bucketsize, elements);
                elements = new HashSet<>();
            }
            counter++;
            elements.add(key);
        }
        // Put whatever remains
        map.put((counter / bucketsize) + 1, elements);
        bucketmap = map;

    }

    public void loadFile() {
        readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt");
    }


    public static void main(String [] args) {
        CrossTrainData ctd = new CrossTrainData();
        ctd.loadFile();

        for (int key : ctd.bucketmap.keySet()){
            System.out.println(key);
            System.out.println(ctd.bucketmap.get(key));
        }
    }

}
