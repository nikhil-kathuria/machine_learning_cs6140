package neu.ml.assignment6;

/**
 * Created by nikhilk on 12/4/15.
 */

import java.util.*;

class KernelCache{
    public double k11;
    public double k12;
    public double k22;

    public KernelCache(double k11, double k12, double k22){
        this.k11 = k11;
        this.k12 = k12;
        this.k22 = k22;
    }
}

public class SMOSolver {
    ArrayList<Integer> ids;

    double [][] train;
    double [] labels;
    double [] alphas;
    double [] weights;

    // Map for rowID -> Error for non bound Alpha's
    HashMap<Integer, Double> cache;
    double [] cached;

    // Set for Storing rows, where Alpha is not bounded i.e. between 0 and C
    HashSet<Integer> nonBounded;

    double tol = 0.01;
    double C = .01;
    double epsilon = 0.001;
    double B;

    // Kernel Cache
    KernelCache KC;

    public SMOSolver(double [][] train, double[] labels){
        this.train = train;
        this.labels = labels;

        initialize();
    }

    public void initialize(){
        KC = new KernelCache(0,0,0);

        ids = new ArrayList<Integer>(labels.length);
        alphas = new double[labels.length];
        cached = new double[labels.length];

        nonBounded = new HashSet<>();
        weights = new double[train[0].length];
        B = 0;


        for (int itr =0; itr < labels.length; itr ++){
            ids.add(itr);
            alphas[itr] = 0;
            cached[itr] = 0;
        }

        for (int itr =0; itr < weights.length; itr++){
            weights[itr] = 0;
        }

        updateCache();

    }

    public double computeL(double alpha1, double alpha2, double l1, double l2){
        if (l1 != l2) {
            return Math.max(0, alpha2 - alpha1);
        } else
            return Math.max(0, alpha2 + alpha1 - C);
    }

    public double computeH(double alpha1, double alpha2, double l1, double l2 ){
        if (l1 != l2){
            return Math.max(C, C + alpha2 - alpha1);
        } else {
            return Math.max(C, alpha2 + alpha1);
        }
    }

    public double svmOut(int row){
        double Fx = 0;
        for (int itr=0; itr < weights.length; itr++){
            Fx += weights[itr] * train[row][itr];
        }

        return Fx + B;
    }

    public double computeKernel(int i1, int i2){
        double sum=0;
        for (int col =0; col < train[0].length; col++ ){
            sum += train[i1][col] * train[i2][col];
        }

        return  sum;
    }


    public boolean noCnZero(double [] arr){
        for (double anArr : arr) {
            if (anArr == 0 || anArr == C) {
                return true;
            }
        }
        return false;
    }


    public double computeETA(int i1, int i2){
        double eta;

        double k11 = 0;
        double k12 = 0;
        double k22 = 0;

        for(int col=0; col< train[0].length; col++){
            k11 = k11 + train[i1][col] * train[i1][col];
            k12 = k12 + train[i1][col] * train[i2][col];
            k22 = k22 + train[i2][col] * train[i2][col];

        }
        KC.k11 = k11;
        KC.k12 = k12;
        KC.k22 = k22;

        eta = k11 + k22 - (2 * k12);
        return eta;
    }


    public void updateWeights(double alpha1, double alpha2, double y1, double y2, int i1, int i2){
        // Check update condition
        double x1 = (alpha1 - alphas[i1]) * y1;
        double x2 = (alpha2 - alphas[i2]) * y2;

        for (int col=0; col < weights.length; col++){
            weights[col] = weights[col] + x1 * train[i1][col] + x2 * train[i2][col];
        }
    }


    public void updateThreshold(double alpha1, double alpha2, double y1, double y2, int i1, int i2){
        double b1;
        double b2;

        b1 = B + cached[i1] +  labels[i1] * (alpha1 - alphas[i1]) * KC.k11 + labels[i2] * (alpha2 - alphas[i2]) * KC.k12;

        b2 = B + cached[i2] + labels[i1] * (alpha1 - alphas[i1]) * KC.k12 + labels[i2] * (alpha2 - alphas[i2]) * KC.k22;

        if ( (alpha1 > 0 && alpha1 < C) && (alpha2 > 0 && alpha2 < C)){
            B = (b1 + b2) / 2;

        } else if(alpha1 > 0 && alpha1 < C)
            B = b1;
        else {
            B = b2;
        }
    }


    public void updateCache(){
        double error;
        for (int row =0; row < alphas.length ; row++){
            error =0;
            for (int col=0; col < weights.length ; col++){
                error += weights[col] * train[row][col];
            }
            cached[row] = error -labels[row];
        }
    }

    public void updateAlphas(double alpha1, double alpha2, int i1, int i2){
        // Update the Non Bounded Alpha set
        if (alpha1 > 0 && alpha1 < C){ nonBounded.add(i1) ;}
        else {nonBounded.remove(i1);}

        if (alpha1 > 0 && alpha1 < C){ nonBounded.add(i1) ;}
        else {nonBounded.remove(i1);}

        // Update the value of Alpha
        alphas[i1] = alpha1;
        alphas[i2] = alpha2;
    }


    public int alpha1Heuristic(int i1){
        // Maximize |E1 - E2| from equation 16
        double E1 = cached[i1];
        double max = Double.MIN_VALUE;

        double abs_error;
        int idx = i1;

        for (int i2 : nonBounded){
            abs_error = Math.abs(E1 - cached[i2]);
            if (i1 == i2) {continue;}

            if (abs_error > max){
                max = abs_error;
                idx = i2;
            }

        }

        return idx;
    }

    public void updateALL(double alpha1, double alpha2, double y1, double y2, int i1, int i2){
        updateThreshold(alpha1, alpha2, y1, y2, i1, i2);
        updateWeights(alpha1, alpha2, y1, y2, i1, i2);
        updateCache();
        updateAlphas(alpha1, alpha2, i1, i2);

    }

    public int takeStep(int i1, int i2){
        if (i1 == i2) {return 0;}
        double alpha1 = alphas[i1];
        double alpha2 = alphas[i2];

        double y1 = labels[i1];
        double y2 = labels[i2];

        double E1 = cached[i1];
        double E2 = cached[i2];

        double s = y1 * y2;

        double L = computeL(alpha1, alpha2, y1, y2);
        double H = computeH(alpha1, alpha2, y1, y2);

        if (L == H){ return 0 ;}

        double eta = computeETA(i1, i2);

        double a2;
        double a1;


        if(eta > 0){
            a2 = alpha2 +  y2 * (E1 - E2) / eta;
            if (a2 < L) {a2 = L ;}
            else if(a2 > H) {a2 = H ;}

        } else {
            // Deviation from FUll SMO
            return 0;


        }

        if(Math.abs(a2 -alpha2) < (epsilon * (a2 + alpha2 + epsilon))){
            return 0;
        }

        a1 = alpha1 + s * (alpha2 - a2);
        updateALL(alpha1, alpha2, y1, y2, i1, i2);
        // Update b values

        // update weight vector

        // update error cache with new lagrange multipliers

        // update alphas with a1 and a2
        return 1;
    }



    public int examineExample(int i2){
        
        double y2 = labels[i2];
        double alpha2 = alphas[i2];
        int i1;

        // For the points not bound i.e between 0 and C
        double E2 = cached[i2];

        double r2 = E2 * y2;

        if((r2 < -tol && alpha2 < C) || (r2 > tol && alpha2 >0)){
            if (noCnZero(alphas)){
                i1 = alpha1Heuristic(i2);
                if (takeStep(i1, i2) == 1) {return  1;}
            }

            // Randomize the KeySet from HashSet
            LinkedList<Integer> myList = new LinkedList<>(nonBounded);
            Collections.shuffle(myList);
            for (int key : myList){
                i1 = key;
                if (takeStep(i1, i2) == 1) {return  1;}
            }

            // Iterate over all the possible i1
            myList = new LinkedList<>(ids);
            Collections.shuffle(myList);
            for (int key :myList ){
                i1 = key;
                if (takeStep(i1, i2) == 1) {return  1;}

            }
        }


        return 0;
    }

    public void Solver(){
        int numchange = 0;
        boolean examineall = true;
        int numPasses =0;

        while(numchange > 0 || examineall ){
            numchange = 0;

            if(examineall){
                for (int itr=0; itr< labels.length; itr++){
                    numchange += examineExample(itr);
                    System.out.println(numchange);
                }

            } else {
                for (int val : nonBounded) {
                    numchange += examineExample(val);
                }
            }

            if (examineall) {
                examineall = false;
            } else if (numchange == 0){
                numPasses++;
                examineall = true;
            }

            //System.out.println(numPasses);
            System.out.println(Arrays.toString(weights));
        }

    }



    public static void main(String [] args){

    }
}
