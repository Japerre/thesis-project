package sampling.strategies;

import org.deidentifier.arx.Data;
import org.deidentifier.arx.DataHandle;
import org.deidentifier.arx.DataSubset;

public interface SamplerStrategy {

    /**
     * creates sample on input data
     * @param data the input data
     * @return the sample
     */
    default DataSubset createSample(Data data){
        DataHandle input = data.getHandle();
        return createSample(input);
    }

    String getPreName();

    /**
     * creates sample on output data
     * @param data the output data
     * @return the sample
     */
    DataSubset createSample(DataHandle output);

    String getPostName();

}
