package neu.ml.assignment1;

/**
 * Created by nikhilk on 9/13/15.
 */

import org.ejml.simple.SimpleMatrix;
import org.ejml.equation.Equation;

import java.util.HashMap;

public class NormalizeData {
    public SimpleMatrix normalize(SimpleMatrix mat){
        SimpleMatrix column;
        Equation eq = new Equation();
        for (int col = 0; col < mat.numCols() - 1 ; col++){
            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;
            double current;

            column = mat.extractVector(false, col);
            for(int row=0; row < mat.numRows(); row++){
                current = mat.get(row, col);
                if(current < min){
                    min = current;
                } else if (current >= max){
                    max = current;
                }
            }
            mat.insertIntoThis(0, col, column.minus(min).divide((max - min)));
        }
        return mat;
    }


    public SimpleMatrix normalizeall(SimpleMatrix mat){
        SimpleMatrix column;
        Equation eq = new Equation();
        for (int col = 0; col < mat.numCols() ; col++){
            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;
            double current;

            column = mat.extractVector(false, col);
            for(int row=0; row < mat.numRows(); row++){
                current = mat.get(row, col);
                if(current < min){
                    min = current;
                } else if (current >= max){
                    max = current;
                }
            }
            mat.insertIntoThis(0, col, column.minus(min).divide((max - min)));
        }
        return mat;
    }

    public static void main(String[] args){
        ParserAndBuildMatrix pab = new ParserAndBuildMatrix();
        HashMap<Integer, String> data =
                pab.readFile("/Users/nikhilk/Documents/NEU_MSCS/ML/Assignment_1/housing_test.txt");
        pab.buildMatrix(data);

        NormalizeData nd = new NormalizeData();
        SimpleMatrix mat = nd.normalizeall(pab.matrix);

        System.out.println(mat);
    }

    }
