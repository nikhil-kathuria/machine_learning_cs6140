package neu.ml.assignment6;

/**
 * Created by nikhilk on 12/10/15.
 */

//import neu.ml.assignment6.SMOSimplified.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

class KernelStore {
    public double k11;
    public double k12;
    public double k22;

    public KernelStore(double k11, double k12, double k22){
        this.k11 = k11;
        this.k12 = k12;
        this.k22 = k22;
    }
}


public class SMOSimplified {
    double [][] train;
    double [] labels;
    double [] alphas;
    KernelStore KC;
    double B;
    Random rand;

    //ArrayList<Integer> mylist;

    double tol = 0.01;
    double C = .01;
    //double epsilon = 0.001;


    SMOSimplified(double [][] train, double[] labels){
        this.train = train;
        this.labels = labels;

        initialize();
    }



    public void initialize(){
        KC = new KernelStore(0,0,0);
        alphas = new double[labels.length];
        B = 0;

        for (int itr =0; itr < labels.length; itr ++){
            alphas[itr] = 0;
        }

        rand = new Random();
    }


    public double computeKernel(int i1, int i2){
        double sum=0;
        for (int col =0; col < train[0].length; col++ ){
            sum += train[i1][col] * train[i2][col];
        }

        return  sum;
    }

    public double svmOut(int row){
        double Fx = 0;
        for (int itr=0; itr < labels.length; itr++){
            Fx += alphas[itr] * labels[itr] * computeKernel(row, itr);
        }

        return Fx + B;
    }


    public int getJ(int I){
        while(true){
            int itr = rand.nextInt((labels.length));
            if (itr != I){
                return itr;
            }
        }
    }

    public double computeL(double alphai, double alphaj, int i, int j){
        if (labels[i] != labels[j]) {
            return Math.max(0, alphaj - alphai);
        } else
            return Math.max(0, alphai + alphaj - C);
    }

    public double computeH(double alphai, double alphaj, int i, int j ){
        if (labels[i] != labels[j]){
            return Math.min(C, C + alphaj - alphai);
        } else {
            return Math.min(C, alphai + alphaj);
        }
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

        eta = 2 * k12 - k11 - k22;
        return eta;
    }

    public void updateThreshold(double ai, double aj, double oldi, double oldj, int i, int j, double Ei, double Ej){
        double b1;
        double b2;

        double diffi = ai - oldi;
        double diffj = aj - oldj;

        b1 = B - Ei -  labels[i] * diffi * KC.k11 - labels[j] * diffj * KC.k12;

        b2 = B - Ej - labels[i] * diffi * KC.k12 - labels[j] * diffj * KC.k22;

        if ( (ai > 0 && ai < C) && (aj > 0 && aj < C)){
            B = (b1 + b2) / 2;

        } else if(ai > 0 && ai < C)
            B = b1;
        else if (aj > 0 && aj < C) {
            B = b2;
        }

        System.out.println(ai + " " + aj + " " + b1 + " " +b2);
    }


    public void runSMO(){
        int maxpass = 3;
        int passes = 0;
        int numchange;
        double oldi;
        double oldj;

        while (passes < maxpass){
            numchange = 0;

            for (int i= 0; i < labels.length ; i++){
                double Ei = svmOut(i) - labels[i];


                if ( (Ei * labels[i] < -tol && alphas[i] < C) || (Ei * labels[i] > tol && alphas[i] > 0)){
                    int j = getJ(i);


                    double Ej = svmOut(j) - labels[j];
                    oldi = alphas[i];
                    oldj = alphas[j];

                    double L = computeL(oldi, oldj, i, j);
                    double H = computeH(oldi, oldj, i, j);

                    if (L == H) {continue;}

                    double eta =  computeETA(i,j);

                    if (eta >= 0) {
                        // Since derivative is negative. So continue
                        System.out.println("Negative Derivative continuing ");
                        continue;
                    }

                    //System.out.println(oldj + " " + labels[j] + " " +  Ei  + " " + Ej  +" " + eta);
                   // System.out.println(Ei - Ej);
                    double aj = oldj - ((labels[j] * (Ei - Ej) ) / eta);
                    //System.out.println(aj);


                    if (aj < L) {aj = L ;}
                    else if(aj > H) {aj = H ;}


                    if (Math.abs(aj - oldj) < 0.00001) {
                        // System.out.println("Negative difference small continuing ");
                        continue;
                    }


                    // Equation 16
                    double ai = oldi + labels[i] * labels[j]  * (oldj - aj);

                    updateThreshold(ai, aj, oldi, oldj, i, j, Ei, Ej);

                    numchange++;
                }
            }

            System.out.println(numchange);
            if (numchange == 0){
                passes++;
            } else {
                passes = 0;
            }
        }

    }


    public void accCalc(double [] labels){
        int hits = 0;
        double Fx = 0;
        for (int row=0; row < labels.length; row ++){
            Fx = svmOut(row);
            //System.out.println(" Prediction " + Fx);
            if (Fx >= 0 ){
                if (labels[row] == 1){
                    hits ++;
                }
            } else {
                if (labels[row] == 0){
                    hits ++;
                }
            }
        }
        System.out.println("Accuracy " +  (double) hits / labels.length);
    }



}


