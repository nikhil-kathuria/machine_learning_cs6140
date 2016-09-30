package neu.ml.assignment1;

import org.ejml.simple.SimpleMatrix;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by nikhilk on 9/16/15.
 */
public class ExecuteClassification {

    public DecisionTree genTree()  {
        // Parse and Build Matrix
        ParserAndBuildMatrix pab = new ParserAndBuildMatrix();
        //HashMap<Integer, String> data = pab.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/housing_train.txt");
        HashMap<Integer, String> data =
                pab.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/spambase.data.txt");
        pab.buildMatrix(data);

        // Build a decision tree
        DecisionTree tree = new DecisionTree(new HashSet<Integer>(0), pab.labelmap, pab.matrix, false);
        DecisionTree head = tree.startBuild(pab, false);

        return head;

        // Print some features and values
        //System.out.println(head.dataset);
        //System.out.println(head.feature);
        //System.out.println(head.value);

    }

    public static void main(String[] args){
        ExecuteClassification exc = new ExecuteClassification();
        exc.genTree();
    }

}

