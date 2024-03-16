package sampling;

import sampling.strategies.BalancedStratifiedSampler;
import sampling.strategies.RandomSampler;
import sampling.strategies.SamplerStrategy;
import sampling.strategies.StratifiedSampler;

public enum Samplers {
    SSAMPLE, RSAMPLE, BSAMPLE;


    public SamplerStrategy getSamplerStrategy(double sampleRate, String target) {
        switch (this) {
            case SSAMPLE:
                return new StratifiedSampler(sampleRate);
            case RSAMPLE:
                return new RandomSampler(sampleRate);
            case BSAMPLE:
                return new BalancedStratifiedSampler(sampleRate, target);
            default:
                throw new IllegalArgumentException("unsupported sampler type: " + this);
        }
    }

    public String getSamplerFolderName() {
        switch (this) {
            case SSAMPLE:
                return "SSAMPLE";
            case RSAMPLE:
                return "RSAMPLE";
            case BSAMPLE:
                return "BSAMPLE";
            default:
                throw new IllegalArgumentException("unsupported sampler type: " + this);
        }
    }

}
