package neu.ml.assignment1;


import java.util.ArrayList;
import java.util.HashMap;
import org.ejml.simple.SimpleMatrix;
/**
 * Created by nikhilk on 9/15/15.
 */


public class PredictRegression {


    public double[] predictClass(DecisionTree tree, SimpleMatrix mat){
        double[] labels = new double[mat.numRows()];
        for(int itr= 0; itr < mat.numRows(); itr++) {
            SimpleMatrix row =  mat.extractVector(true, itr);
            labels[itr] = reachClass(tree, row);
        }
        return labels;
    }

    public double reachClass(DecisionTree tree, SimpleMatrix row){
        // When both of left and right are not null
        DecisionTree node = tree;
        while(node.left != null && node.right != null){
            double val = row.get(0, (node.feature));
            if (val > node.threshold ) {
                //System.out.println(val + "  " + node.threshold);
                node = node.right;
            } else if (val < node.threshold) {
                //System.out.println(val + "  " + node.threshold);
                node = node.left;
            } else {
                //System.out.println(val + "  " + node.threshold);
                return node.prediction;
            }
        }
        // Return the value of the tree node reached last
        return node.prediction;
    }

    public void parseAndPredictTest(boolean tree){
        ParserAndBuildMatrix pab = new ParserAndBuildMatrix();
        HashMap<Integer, String> data =
                pab.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/housing_test.txt");
        pab.buildMatrix(data);

        HashMap<Integer, Double> labelmap = pab.labelmap;

        double[] labels;

        if (tree) {
            ExecuteRegression exreg = new ExecuteRegression();
            DecisionTree head = exreg.genTree();

            labels = predictClass(head, pab.matrix);

        } else {
            ExecuteRegression exreg = new ExecuteRegression();
            SimpleMatrix weights = exreg.normalequation();

            // Create Bias column and add to column
            SimpleMatrix oldmat = pab.matrix;
            SimpleMatrix bias = new SimpleMatrix(oldmat.numRows(), 1);
            bias.set(1.0);
            SimpleMatrix mat = bias.combine(0, bias.numCols(), oldmat);

            labels = mat.mult(weights).getMatrix().getData();
        }


        double mse = 0;

        for (int itr= 0 ; itr < labels.length ; itr++) {
            System.out.println(labelmap.get(itr) + " " + labels[itr]);
            mse += Math.pow( (labelmap.get(itr) - labels[itr]), 2);
        }

        mse = mse / labels.length;

        System.out.println("MSE " + mse );
    }

    public static void main(String [] args) {
        PredictRegression pre = new PredictRegression();
        pre.parseAndPredictTest(true);
    }
}
