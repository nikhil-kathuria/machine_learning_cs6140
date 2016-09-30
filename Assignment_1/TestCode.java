package neu.ml.assignment1;

/**
 * Created by nikhilk on 9/14/15.
 */

import java.util.*;
import java.util.Random;

public class TestCode {

    public static void main(String[] args) {
        String str = "Pekkabu";

        Set<Integer> s1 = new HashSet<Integer>();
        Set<Integer> s2 = new HashSet<Integer>();
        s1.add(1);
        s1.add(3);
        s1.add(4);
        s1.add(5);

        /**
        for ( int s : s1){
            System.out.println(s);
        }

        for ( int s : s1){
            System.out.println(s);
        }

        for ( int s : s1){
            System.out.println(s);
        }
        **/

        HashSet<Integer> hash = new HashSet<>();
        hash.add(1);
        hash.add(2);
        hash.add(3);
        hash.add(4);

        ArrayList<Integer> arl = new ArrayList<>(hash);

        Double probone = (double) 8/ 10;

        Double probzero = (double) 5 / 10;

        System.out.println(probone.compareTo(probzero));

        System.out.println(str.subSequence(0, str.length() - 1));


        double val = probone * (Math.log(probone) / Math.log(2));
                //+ probzero * (Math.log(probzero) / Math.log(2));

        /**Random rand = new Random();
        int count=20;

        while (count > 0){
            count--;
            int randomNum = rand.nextInt((5 - 0) + 1) + 0;
            System.out.println(randomNum);
        }**/
    }
}
