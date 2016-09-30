package neu.ml.assignment1;

import java.util.HashMap;
import java.util.HashSet;
import org.ejml.simple.SimpleMatrix;

/**
 * Created by nikhilk on 9/16/15.
 */

public class ExecuteRegression {
    public DecisionTree genTree() {

        // Parse and Build Matrix
        ParserAndBuildMatrix pab = new ParserAndBuildMatrix();
        HashMap<Integer, String> data =
                pab.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/housing_train.txt");
        pab.buildMatrix(data);

        // Build a decision tree
        DecisionTree tree = new DecisionTree(new HashSet<Integer>(0), pab.labelmap, pab.matrix,true);
        DecisionTree head = tree.startBuild(pab, true);

        // Print some features and values
        //System.out.println(head.dataset);
        //System.out.println(head.feature);
        //System.out.println(head.value);

        return  head;

    }

    public SimpleMatrix normalequation(){
        ParserAndBuildMatrix pab = new ParserAndBuildMatrix();
        HashMap<Integer, String> data =
                pab.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/housing_train.txt");
        pab.buildMatrix(data);

        HashMap<Integer, Double> map = pab.labelmap;
        SimpleMatrix labels = new SimpleMatrix(map.keySet().size(), 1);

        for (int itr= 0; itr < labels.numRows(); itr++) {
            labels.set(itr,0, map.get(itr));
        }


        // Create Bias column
        SimpleMatrix oldmat = pab.matrix;
        SimpleMatrix bias = new SimpleMatrix(oldmat.numRows(), 1);
        bias.set(1.0);

        // Add bias column
        SimpleMatrix mat = bias.combine(0, bias.numCols(), oldmat);

        SimpleMatrix tran = mat.transpose();

        SimpleMatrix weights = tran.mult(mat).pseudoInverse().mult(tran).mult(labels);

        return weights;
    }


}
