package neu.ml.assignment6;

/**
 * Created by nikhilk on 12/12/15.
 */



import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

/**
 * Created by nikhilk on 10/30/15.
 */
public class ParseSpam {
    //int rows;
    //int columns;
    double[][] data;
    double [] labels;
    String splitString = ",";
    public HashMap<Integer, HashSet<Integer>> bucketmap;
    HashMap<Integer, String> datamap;

    // Other split pattern "\\s+"

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

            /**
            if (! data.isEmpty()) {
                rows = data.size();
                columns = data.get(0).split("\\s+").length;
                columns = data.get(0).split(splitString).length;
            }
             **/

        } catch(Exception e) {
            e.printStackTrace(System.out);
            //System.out.println("Error while reading file line by line: " + e.getMessage());
        }

        genBuckets(new HashSet<Integer>(data.keySet()));
        datamap = data;
        return data;

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

    public void buildData(HashMap<Integer, String> map) {
        int rows = map.size();
        double[] target = new double[rows];
        int columns = columns = map.get(0).split(splitString).length;

        double[][] data2d = new double[rows][columns - 1];
        for (int row = 0 ; row < rows ; row++) {
            String[] datarow = map.get(row).split(splitString);
            for(int col = 0; col < columns - 1; col++) {
                data2d[row][col] = Double.valueOf(datarow[col]);
            }
            target[row] =  Double.valueOf(datarow[columns - 1]);
        }

        data = data2d;
        this.labels = target;

    }

    public void buildData(HashMap<Integer, String> data, HashMap<Integer, HashSet<Integer>> bucketmap,
                            HashSet<Integer> set) {
        int rows = 0;
        HashSet<Integer> rowset = new HashSet<>();
        //System.out.println(set);

        for (int key : set) {
            rowset.addAll(bucketmap.get(key));
            rows += bucketmap.get(key).size();
        }

        int columns = data.get(0).split(",").length;

        double[][] data2d = new double[rows][columns - 1];
        double[] target = new double[rows];
        int rowid = 0;

        for (int row : rowset) {
            String[] datarow = data.get(row).split(",");
            for (int col = 0; col < columns - 1; col++) {
                data2d[rowid][col] = Double.valueOf(datarow[col]);
            }
            target[rowid]= Double.valueOf(datarow[columns - 1]);
            rowid++;
        }
        this.data = data2d;
        this.labels = target;

    }

}


