package sampling.strategies;

import org.deidentifier.arx.DataHandle;
import org.deidentifier.arx.DataSubset;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RandomSampler implements SamplerStrategy {

    private final double sampleSize;

    public RandomSampler(double sampleSize) {
        if (sampleSize < 0 | sampleSize > 1) {
            throw new IllegalArgumentException("Sample size must be in range [0-1].");
        }
        this.sampleSize = sampleSize;
    }

    @Override
    public DataSubset createSample(DataHandle output) {
        int size = output.getNumRows();
        List<Integer> numbers = IntStream.range(0, size).boxed().collect(Collectors.toList());
        Collections.shuffle(numbers);
        Set<Integer> sample = new HashSet<>(numbers.subList(0, (int) (size * sampleSize)));
        //printEqDistribution(output,sample);
        return DataSubset.create(size, sample);
    }

    @Override
    public String getPostName() {
        return "RandomPost(" + sampleSize + ")";
    }

    @Override
    public String getPreName() {
        return "RandomPre(" + sampleSize + ")";
    }

}
