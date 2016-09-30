package neu.ml.assignment1;

import org.ejml.simple.SimpleMatrix;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by nikhilk on 9/17/15.
 */


public class PredictClassification {
    public SimpleMatrix testmat;
    public SimpleMatrix trainmat;

    public HashMap<Integer, Double> testmap;
    public HashMap<Integer, Double> trainmap;

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
        //System.out.println("Node Feature : " + node.feature);
        while(node.left != null && node.right != null){
            //System.out.println("Node Feature : " + node.feature);
            //System.out.println("Node Threshold : " + node.threshold);
            //System.out.println("Node Entropy : " + node.minerror);
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
        //System.out.println("Node Entropy : " + node.minerror);
        return node.prediction;
    }


    public void buildMatrix(HashMap<Integer, String> data, HashMap<Integer, HashSet<Integer>> bucketmap,
                               HashSet<Integer> set, boolean bool) {
        HashMap<Integer, Double> labelmap = new HashMap<Integer, Double>();

        int rows = 0;
        HashSet<Integer> rowset = new HashSet<>();
        //System.out.println(set);
        for(int key : set) {
            rowset.addAll(bucketmap.get(key));
            rows += bucketmap.get(key).size();
        }

        int columns = data.get(0).split(",").length;

        //System.out.println(rows);
        //System.out.println(columns);

        double[][] data2d = new double[rows][columns - 1];
        int rowid = 0;

        for (int row : rowset) {
            String[] datarow = data.get(row).split(",");
            for(int col = 0; col < columns - 1; col++) {
                data2d[rowid][col] = Double.valueOf(datarow[col]);
            }
            labelmap.put(rowid, Double.valueOf(datarow[columns - 1]));
            rowid++;
        }

        if (bool) {
            trainmat = new SimpleMatrix(data2d);
            trainmap = labelmap;
        } else {
            testmat = new SimpleMatrix(data2d);
            testmap = labelmap;
        }

    }


    public double msecalc(double[] labels, HashMap<Integer, Double> testmap){
        double mse = 0;
        for (int itr= 0 ; itr < labels.length ; itr++) {
            //System.out.println(labels[itr] + " " +testmap.get(itr));
            mse += Math.pow( (labels[itr] - testmap.get(itr)), 2);
        }

        mse = mse / labels.length;
        return mse;
    }

    public double acccalc(double[] labels, HashMap<Integer, Double> testmap){
        double acc = 0;
        double tp = 0;
        double fp = 0;
        double tn = 0;
        double fn = 0;

        for (int itr= 0 ; itr < labels.length ; itr++) {
           //System.out.println(labels[itr] + " " + testmap.get(itr));
           if (testmap.get(itr) == labels[itr]) {
               if (labels[itr] == 1){ tp++ ;}
               else {tn++ ;}
           } else {
               if (labels[itr] == 0) { fn++ ;}
               else {fp++ ;}
           }
        }

        System.out.println("True Positive " + tp);
        System.out.println("False Positive " + fp);
        System.out.println("True Negative " + tn);
        System.out.println("False Negative " + fn);
        acc = (tp + tn) /labels.length;
        return acc * 100;
    }


    public SimpleMatrix normalequation(SimpleMatrix oldmat, HashMap<Integer, Double> map){
        SimpleMatrix labels = new SimpleMatrix(map.keySet().size(), 1);

        for (int itr= 0; itr < labels.numRows(); itr++) {
            labels.set(itr,0, map.get(itr));
        }
        // Create and Add Bias column
        SimpleMatrix bias = new SimpleMatrix(oldmat.numRows(), 1);
        bias.set(1.0);
        SimpleMatrix mat = bias.combine(0, bias.numCols(), oldmat);

        SimpleMatrix tran = mat.transpose();
        SimpleMatrix weights = tran.mult(mat).invert().mult(tran).mult(labels);

        return weights;
    }


    public void buildAndPredict(CrossTrainData ctd, boolean tree){

        // Total buckets and tree holder
        HashSet<Integer> total = new HashSet<>(ctd.bucketmap.keySet());
        DecisionTree head;
        double summse = 0;
        double sumacc = 0;
        double acc = 0;
        double mse = 0;


        for (int key : total){
            // Assign test and train. Update train
            HashSet<Integer> train = new HashSet<>(total);
            HashSet<Integer> test = new HashSet<>(key);
            test.add(key);
            train.remove(key);


            buildMatrix(ctd.datamap, ctd.bucketmap, train, true);
            buildMatrix(ctd.datamap, ctd.bucketmap, test, false);

            double[] labels;
            SimpleMatrix weights;

            if (tree) {
                head = new DecisionTree(new HashSet<>(trainmap.keySet()), trainmap, trainmat, false);
                head.buildTree();
                labels =  predictClass(head, testmat);

                acc = acccalc(labels, testmap);

                System.out.println("Accuracy for bucket" +  test + " is " + acc );

                sumacc += acc;


            } else {

                weights = normalequation(trainmat, trainmap);

                // Create and Add Bias column for TESTMAP
                SimpleMatrix bias = new SimpleMatrix(testmat.numRows(), 1);
                bias.set(1.0);
                SimpleMatrix mat = bias.combine(0, bias.numCols(), testmat);

                labels = mat.mult(weights).getMatrix().getData();

                mse = msecalc(labels, testmap);

                System.out.println(" MSE for bucket " + test + " is " + mse );

                summse += mse;

                /**for (int itr = 0; itr < labels.length; itr++) {
                    if (labels[itr] >= 0.0) {
                        labels[itr] = 1.0;
                    } else {
                        labels[itr] = 0.0;
                    }

                }**/
            }
        }

        System.out.println(" Aevrage MSE " + summse / total.size() );
        System.out.println(" Average ACC " + sumacc / total.size() );
    }


    public static void main(String [] args) {
        CrossTrainData ctd = new CrossTrainData();
        ctd.loadFile();

        PredictClassification prc = new PredictClassification();
        prc.buildAndPredict(ctd, true);

    }
}
