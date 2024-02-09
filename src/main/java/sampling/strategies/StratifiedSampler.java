package sampling.strategies;

import org.deidentifier.arx.DataHandle;
import org.deidentifier.arx.DataSubset;
import sampling.Grouper;

import java.util.*;

public class StratifiedSampler implements SamplerStrategy {

    private final double sampleSize;

    public StratifiedSampler(double sampleSize) {
        if (sampleSize < 0 | sampleSize > 1) {
            throw new IllegalArgumentException("Sample size must be in range [0-1].");
        }
        this.sampleSize = sampleSize;
    }

    protected static Map<String,Integer> stratifiedSampleSizes(Grouper groups, double samplePercentage){
        int totalSampleAmount = (int) (samplePercentage * groups.rows);
        // calculate samplesize for each group
        Random rand = new Random();
        Map<String,Integer> sizes = new HashMap<>();
        Map<String,Integer> sampleSizes = new HashMap<>();
        final Integer[] currentSampleAmount = {0};
        groups.equivalenceClasses.forEach((qid,eq) ->{
            sizes.put(qid,eq.size());
            int sampleSize = (int) (eq.size() * samplePercentage);
            sampleSizes.put(qid,sampleSize);
            currentSampleAmount[0] += sampleSize;
        });

        // increase 1-error values cache
        Map<Double,List<String>> errorQIDMap = new HashMap<>();
        Queue<Double> errors = new PriorityQueue<>();
        //precompute increase 1-error
        sampleSizes.forEach((qid, sampleSize) -> {
            double error = (double)(sampleSize+1)/sizes.get(qid) - samplePercentage;
            errorQIDMap.computeIfAbsent(error,k->new ArrayList<>()).add(qid);
            errors.add(error);
        });

        // will always need more records because of int cast
        while (totalSampleAmount != currentSampleAmount[0]){
            // get lowest error and corresponding qids
            double lowestError = errors.poll();
            List<String> smallestQids = errorQIDMap.get(lowestError);
            // take random qid from list
            int index = rand.nextInt(smallestQids.size());
            String randomQID = smallestQids.get(index);
            // add entry from this qid
            int qidSampleSize = sampleSizes.get(randomQID)+1;
            sampleSizes.put(randomQID,qidSampleSize);
            currentSampleAmount[0]++;
            // remove qid from current list in errorMap
            if (smallestQids.size()==1){
                errorQIDMap.remove(lowestError);
            }else{
                smallestQids.remove(index);
            }
            //calculate new error
            double newError = (double) (qidSampleSize+1)/sizes.get(randomQID) - samplePercentage;
            errorQIDMap.computeIfAbsent(newError,k->new ArrayList<>()).add(randomQID);
            errors.add(newError);
        }
        return sampleSizes;
    }

    protected static Map<String,Integer> stratifiedSampleSizesWithTarget(Grouper groups, int totalSampleAmount, String targetValue){
        double samplePercentage = (double)totalSampleAmount/groups.targetCounts.get(targetValue);
        // calculate samplesize for each group
        Random rand = new Random();
        Map<String,Integer> sizes = new HashMap<>();
        Map<String,Integer> sampleSizes = new HashMap<>();
        final Integer[] currentSampleAmount = {0};
        groups.equivalenceClasses.forEach((qid,eq) ->{
            int size = eq.targetCounts.getOrDefault(targetValue,0);
            sizes.put(qid,size);
            int sampleSize = (int) (size * samplePercentage);
            sampleSizes.put(qid,sampleSize);
            currentSampleAmount[0] += sampleSize;
        });

        // increase 1-error values cache
        Map<Double,List<String>> errorQIDMap = new HashMap<>();
        Queue<Double> errors = new PriorityQueue<>();
        //precompute increase 1-error
        sampleSizes.forEach((qid, sampleSize) -> {
            double error = (double)(sampleSize+1)/sizes.get(qid) - samplePercentage;
            errorQIDMap.computeIfAbsent(error,k->new ArrayList<>()).add(qid);
            errors.add(error);
        });

        // will always need more records because of int cast
        while (totalSampleAmount != currentSampleAmount[0]){
            // get lowest error and corresponding qids
            double lowestError = errors.poll();
            List<String> smallestQids = errorQIDMap.get(lowestError);
            // take random qid from list
            int index = rand.nextInt(smallestQids.size());
            String randomQID = smallestQids.get(index);
            // add entry from this qid
            int qidSampleSize = sampleSizes.get(randomQID)+1;
            sampleSizes.put(randomQID,qidSampleSize);
            currentSampleAmount[0]++;
            // remove qid from current list in errorMap
            if (smallestQids.size()==1){
                errorQIDMap.remove(lowestError);
            }else{
                smallestQids.remove(index);
            }
            //calculate new error
            double newError = (double) (qidSampleSize+1)/sizes.get(randomQID) - samplePercentage;
            errorQIDMap.computeIfAbsent(newError,k->new ArrayList<>()).add(randomQID);
            errors.add(newError);
        }
        return sampleSizes;
    }

    @Override
    public DataSubset createSample(DataHandle output) {
        Grouper groups = new Grouper(output);

        Map<String,Integer> sampleSizes = stratifiedSampleSizes(groups,this.sampleSize);
        // take sample
        Set<Integer> sample = new HashSet<>();
        sampleSizes.forEach((qid,sampleSize) -> {
            if (sampleSize > 0) {
                List<Integer> group = groups.get(qid).rows;
                Collections.shuffle(group);
                sample.addAll(group.subList(0, sampleSize));
            }
        });

        //printEqDistribution(output,sample);

        return DataSubset.create(output.getNumRows(),sample);
    }

    @Override
    public String getPostName() {
        return "StratifiedPost(" + sampleSize + ")";
    }

    @Override
    public String getPreName() {
        return "StratifiedPre(" + sampleSize + ")";
    }

}
