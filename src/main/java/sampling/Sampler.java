package sampling;

import org.deidentifier.arx.Data;
import org.deidentifier.arx.DataHandle;
import org.deidentifier.arx.DataSubset;
import sampling.strategies.RandomSampler;
import sampling.strategies.SamplerStrategy;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class Sampler {
    /**
     * Indicates the current preSample set
     */
    private DataSubset preSample;

    /**
     * Indicates the current postSample set
     */
    private DataSubset postSample;

    /**
     * Indicates if the postsample is the same as the presample.
     * Default: true
     */
    private boolean reusePre = false;
    private boolean reuseSample = false;

    // defaults are random 10%
    private SamplerStrategy preSampler = new RandomSampler(0.1);
    private SamplerStrategy postSampler = new RandomSampler(0.1);



    /**
     * copy constructor
     * @param sampler The object to copy
     */
    public Sampler(Sampler sampler){
        this.reusePre = sampler.reusePre;
        this.reuseSample = sampler.reuseSample;
        this.preSampler = sampler.preSampler;
        this.postSampler = sampler.postSampler;
    }

    /**
     * Constructor
     *
     * @param preSampler  The pre sample  strategy to use.
     * @param postSampler The post sample  strategy to use.
     */
    public Sampler(SamplerStrategy preSampler, SamplerStrategy postSampler) {
        this.preSampler = preSampler;
        this.postSampler = postSampler;
    }

    /**
     * Constructor
     *
     * @param sampler The pre and post sampler strategy to use.
     */
    public Sampler(SamplerStrategy sampler) {
        this.preSampler = sampler;
        this.postSampler = sampler;
    }
    /**
     * Creates a pre sample on {@code data} using the {@link Sampler#preSampler}.
     *
     * @param data The input data being sampled.
     * @return
     */
    private DataSubset createPreSample(Data data) {
        if (this.reusePre && postSample != null) {
            this.preSample = this.postSample;
        } else {
            this.preSample = preSampler.createSample(data);
        }
        return this.preSample;
    }

    /**
     * Creates a post sample on {@code output} using the {@link Sampler#postSampler}
     *
     * @param output The output data.
     * @return
     */
    private DataSubset createPostSample(DataHandle output) {
        if (this.reusePre && preSample != null) {
            this.postSample = this.preSample;
        } else {
            this.postSample = postSampler.createSample(output);
        }
        return this.postSample;
    }

    /**
     * Saves {@link Sampler#preSample} if created to {@code path}
     *
     * @param path The path to the file.
     * @throws IOException
     */
    public void savePreSample(String path) throws IOException {
        if (this.preSample == null) return;
        List<String> sampleSorted = Arrays.stream(preSample.getArray()).sorted().boxed().map(Object::toString).collect(Collectors.toList());
        Path file = Paths.get(path);
        Files.write(file, sampleSorted, StandardCharsets.UTF_8);
    }

    /**
     * Saves {@link Sampler#preSample} in {@code path/samplerName/sample.csv}
     *
     * @param path The path to the folder.
     * @throws IOException
     */
    public void savePreSample(Path path) throws IOException {
        this.savePreSample(path.resolve(preSampler.getPreName()).resolve("sample.csv").toString());
    }

    /**
     * Saves {@link Sampler#postSample} if created to {@code path}
     *
     * @param path The path to the file.
     * @throws IOException
     */
    public void savePostSample(String path) throws IOException {
        if (this.postSample == null) return;
        List<String> sampleSorted = Arrays.stream(postSample.getArray()).sorted().boxed().map(Object::toString).collect(Collectors.toList());
        Path file = Paths.get(path);
        Files.write(file, sampleSorted, StandardCharsets.UTF_8);
    }

    /**
     * Saves {@link Sampler#postSample} in {@code path/samplerName/sample.csv}
     *
     * @param path The path to the folder.
     * @throws IOException
     */
    public void savePostSample(Path path) throws IOException {
        this.savePostSample(path.resolve(postSampler.getPostName()).resolve("sample.csv").toString());
    }

    /**
     * Loads the {@link Sampler#preSample} from the file at {@code path} containing the sample row numbers.
     *
     * @param data The data to create the sample on.
     * @param path The path to the file.
     * @return {@link Sampler#preSample}
     * @throws IOException
     */
    public DataSubset readPreSample(Data data, String path) throws IOException {
        Path file = Paths.get(path);
        Set<Integer> sample = Files.readAllLines(file, StandardCharsets.UTF_8).stream().map(Integer::valueOf).collect(Collectors.toSet());
        this.preSample = DataSubset.create(data, sample);
        return preSample;
    }

    /**
     * Loads the {@link Sampler#postSample} from the file at {@code path} containing the sample row numbers.
     *
     * @param data The data to create the sample on.
     * @param path The path to the file.
     * @return {@link Sampler#postSample}
     * @throws IOException
     */
    public DataSubset readPostSample(Data data, String path) throws IOException {
        Path file = Paths.get(path);
        Set<Integer> sample = Files.readAllLines(file, StandardCharsets.UTF_8).stream().map(Integer::valueOf).collect(Collectors.toSet());
        this.postSample = DataSubset.create(data, sample);
        return postSample;
    }

    /**
     * Sets if {@link Sampler#postSample} must equal the used {@link Sampler#preSample}.
     *
     * @param reusePre Default {@code true}
     */
    public void setReusePre(boolean reusePre) {
        this.reusePre = reusePre;
    }

    /**
     * Get the {@link Sampler#preSample}. Creates a new one if {@link #reuseSample} is {@code false} or {@link #preSample} is unset.
     *
     * @param data The data to create the sample on.
     * @return {@link Sampler#preSample}
     * @see #createPreSample(Data)
     */
    public DataSubset getPreSample(Data data) {
        if (preSample == null || !reuseSample) {
            preSample = createPreSample(data);
        }
        return preSample;
    }

    /**
     * Get the {@link Sampler#postSample}. Creates a new one if {@link #reuseSample} is {@code false} or {@link #postSample} is unset.
     *
     * @param output The output data.
     * @return {@link Sampler#postSample}
     * @see #createPostSample(DataHandle)
     */
    public DataSubset getPostSample(DataHandle output) {
        if (postSample == null || !reuseSample) {
            postSample = createPostSample(output);
        }
        return postSample;
    }

    public void setPreSampler(SamplerStrategy preSampler) {
        this.preSampler = preSampler;
    }

    public void setPostSampler(SamplerStrategy postSampler) {
        this.postSampler = postSampler;
    }

    public String getPreSamplerName() {
        return this.preSampler.getPreName();
    }

    public String getPostSamplerName() {
        return this.postSampler.getPostName();
    }

    /**
     * Sets if calling {@link #getPostSample(DataHandle)} and {@link #getPreSample(Data)} will always return the same already created sample.
     *
     * @param reuseSample
     */
    public void setReuseSample(boolean reuseSample) {
        this.reuseSample = reuseSample;
    }
}
