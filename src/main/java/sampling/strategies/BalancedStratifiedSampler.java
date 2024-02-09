package sampling.strategies;

import org.deidentifier.arx.DataHandle;
import org.deidentifier.arx.DataSubset;
import sampling.Grouper;

import java.util.*;
import java.util.Map.Entry;

import static io.Utils.spreadEqually;

public class BalancedStratifiedSampler implements SamplerStrategy {

    private final double sampleSize;
    private final String target;

    public BalancedStratifiedSampler(double sampleSize, String target) {
        if (sampleSize < 0 | sampleSize > 1) {
            throw new IllegalArgumentException("Sample size must be in range [0-1].");
        }
        this.sampleSize = sampleSize;
        this.target = target;
    }

    @Override
    public DataSubset createSample(DataHandle output) {
        if(output.getColumnIndexOf(target)==-1){
            System.err.println("Target: " + target + " ,not available in dataset.");
            return null;
        }

        // create grouping
        int sampleAmount = (int) (sampleSize * output.getNumRows());
        Grouper eqGroups = new Grouper(output,this.target);

        // calculate for each group if the target is possible, if not adjust target and spread leftover equally
        // assume same order as entrylist
        List<Entry<String,Integer>> entryList = new ArrayList<>(eqGroups.targetCounts.entrySet());
        Collections.shuffle(entryList);
        entryList.sort(Entry.comparingByValue());

        List<Integer> targetAmounts = spreadEqually(sampleAmount, eqGroups.targetGroups.size());
        Map<String,Integer> amountsForTarget = new HashMap<>();
        for(int i = 0;i < eqGroups.targetGroups.size();i++) {
            if (targetAmounts.get(i) > entryList.get(i).getValue()) {
                int difference = targetAmounts.get(i) - entryList.get(i).getValue();
                targetAmounts.set(i, entryList.get(i).getValue());
                int leftoverGroups = eqGroups.targetGroups.size() - i - 1;
                List<Integer> redistributed = spreadEqually(difference, leftoverGroups);
                for (int j = 0; j < redistributed.size(); j++) {
                    targetAmounts.set(i + 1 + j, targetAmounts.get(i + 1 + j) + redistributed.get(j));
                }
            }
            amountsForTarget.put(entryList.get(i).getKey(),targetAmounts.get(i));
        }

        Set<Integer> sample = new HashSet<>();
        // stratified sample for each target value
        amountsForTarget.forEach((targetValue, targetAmount) -> {
            Map<String, Integer> eqSampleSizes = StratifiedSampler.stratifiedSampleSizesWithTarget(eqGroups,targetAmount,targetValue);
            // take the sample
            eqSampleSizes.forEach((qid, amount) -> {
                if (amount>0) {
                    List<Integer> rows = eqGroups.get(qid).rowsByTarget.get(targetValue);
                    Collections.shuffle(rows);
                    sample.addAll(rows.subList(0, amount));
                }
            });
        });

        return DataSubset.create(output.getNumRows(),sample);
    }

    @Override
    public String getPostName() {
        return "BalancedPost(" + sampleSize + ")";
    }

    @Override
    public String getPreName() {
        return "BalancedPre(" + sampleSize + ")";
    }

}
