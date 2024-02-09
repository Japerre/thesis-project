package sampling;

import sampling.strategies.BalancedStratifiedSampler;
import sampling.strategies.RandomSampler;
import sampling.strategies.SamplerStrategy;
import sampling.strategies.StratifiedSampler;

public enum Samplers {
    SSAMPLE, RSample, BSample;

    public SamplerStrategy getSamplerStrategy(double sampleRate, String target) {
        switch (this) {
            case SSAMPLE:
                return new StratifiedSampler(sampleRate);
            case RSample:
                return new RandomSampler(sampleRate);
            case BSample:
                return new BalancedStratifiedSampler(sampleRate, target);
            default:
                throw new IllegalArgumentException("unsupported sampler type: " + this);
        }
    }

    public String getSamplerFolderName() {
        switch (this) {
            case SSAMPLE:
                return "SSAMPLE";
            case RSample:
                return "RSample";
            case BSample:
                return "BSample";
            default:
                throw new IllegalArgumentException("unsupported sampler type: " + this);
        }
    }

}
