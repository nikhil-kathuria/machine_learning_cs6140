package neu.ml.assignment6;

/**
 * Created by nikhilk on 12/14/15.
 */



public class Normalize {
    // Normalize the data except the last column
    public double[][] normalize(double[][] data) {
        for (int col = 0; col < data[0].length - 1; col++) {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            double current;
            double denominator;


            for (int row = 0; row < data.length; row++) {
                current = data[row][col];
                if (current < min) {
                    min = current;
                } else if (current >= max) {
                    max = current;
                }
            }

            if (max == min) {
                denominator = 1;
            } else {
                denominator = max - min;
            }
            for (int row = 0; row < data.length; row++) {
                data[row][col] = (data[row][col] - min) / denominator;
            }
        }
        return data;
    }

    // Normalize all columns of data
    public double[][] normalizeall(double[][] data) {
        for (int col = 0; col < data[0].length; col++) {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            double current;
            double denominator;


            for (int row = 0; row < data.length; row++) {
                current = data[row][col];
                if (current < min) {
                    min = current;
                } else if (current >= max) {
                    max = current;
                }
            }

            if (max == min) {
                denominator = 1;
            } else {
                denominator = max - min;
            }
            for (int row = 0; row < data.length; row++) {
                data[row][col] = (data[row][col] - min) / denominator;
            }
        }
        return data;
    }

    public static void main(String [] args){

    }
}