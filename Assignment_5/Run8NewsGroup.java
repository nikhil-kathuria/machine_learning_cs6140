package neu.ml.assignment5;

import java.util.HashMap;

/**
 * Created by nikhilk on 11/20/15.
 */
public class Run8NewsGroup {
    int rows;
    int columns = 1754;
    public String splitstring = "\\s+";
    public String wordsplit = ":";

    double [] labels;
    double [][] data;


    public void RunEcoc(){
        ParseData pd = new ParseData();
        HashMap<Integer, String> mymap = pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_4/ECOC/8newsgroup/train.trec/feature_matrix.txt");
        buildData(mymap);
        System.out.println(mymap.size());

        double [] trainlabels = labels;
        double [][] train = data;

        mymap = pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_4/ECOC/8newsgroup/test.trec/feature_matrix.txt");
        buildData(mymap);
        System.out.println(mymap.size());



        EcocBoosting eb = new EcocBoosting();
        eb.setData(train, trainlabels, data, labels, 200);

        //labels = null; data= null; trainlabels =null; train = null; mymap=null;

        // Generate Ecoc table
        eb.table =  eb.genEcocTable(8, 20);
        eb.genLabelFunctions(eb.table, eb.trainlabels);

        //Perform boosting, make predictions,
        eb.runBoosting();
        double [] predictions = eb.makePredictions();

        eb.accCalc(eb.testlabels, predictions);
    }




    public void buildData(HashMap<Integer, String> map){
            String [] line;
            labels = new double[map.size()];
            data = new double[map.size()][columns];
            //System.out.println(map.get(0));

            for(int row = 0 ; row < map.size() ;row++ ){
                String str = map.get(row);
                line = str.split(splitstring);
                String [] word;

                for (int col =0 ; col < line.length; col++){
                    if(col == 0){
                        labels[row] = Double.valueOf(line[col]);
                    } else {
                        word = line[col].split(wordsplit);
                        data[row][Integer.valueOf(word[0])] = Double.valueOf(word[1]);
                    }
                }
            }

        }


    public static void main(String [] args){
        Run8NewsGroup rng = new Run8NewsGroup();
        rng.RunEcoc();
    }



}
