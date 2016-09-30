package neu.ml.assignment6;

import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.Feature;
import neu.ml.assignment5.ParseData;


/**
 * Created by nikhilk on 11/26/15.
 */
public class DigitsLinear {

    public void runLinear(){
        ParseData pd = new ParseData();
        double [][] train = pd.populateData(pd.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htrain.txt"));
        double [] trainlabels = pd.populateLabels("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_5/HF/htrainlabels.txt");

        Problem prob = new Problem();
        prob.l = trainlabels.length;
        prob.n = train[0].length;
        prob.y = trainlabels;


    }
}

