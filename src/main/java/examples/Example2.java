package examples;

import org.deidentifier.arx.*;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.metric.Metric;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class Example2 extends Example {

    /**
     * Entry point.
     *
     * @param args the arguments
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {

        Data data = Data.create("data/ARX_examples_data/test.csv", StandardCharsets.UTF_8, ';');

        // Define input files
        data.getDefinition().setAttributeType("age", AttributeType.Hierarchy.create("data/ARX_examples_data/test_hierarchy_age.csv", StandardCharsets.UTF_8, ';'));
        data.getDefinition().setAttributeType("gender", AttributeType.Hierarchy.create("data/ARX_examples_data/test_hierarchy_gender.csv", StandardCharsets.UTF_8, ';'));
        data.getDefinition().setAttributeType("zipcode", AttributeType.Hierarchy.create("data/ARX_examples_data/test_hierarchy_zipcode.csv", StandardCharsets.UTF_8, ';'));

        // Create an instance of the anonymizer
        ARXAnonymizer anonymizer = new ARXAnonymizer();

        // Execute the algorithm
        ARXConfiguration config = ARXConfiguration.create();
        config.addPrivacyModel(new KAnonymity(2));
        config.setSuppressionLimit(0d);
        config.setQualityModel(Metric.createLossMetric(0.5, Metric.AggregateFunction.ARITHMETIC_MEAN));
        ARXResult result = anonymizer.anonymize(data, config);

        // Print info
        printResult(result, data);

        // Write results
        System.out.print(" - Writing data...");
        result.getOutput(false).save("data/test_anonymized.csv", ';');
        System.out.println("Done!");
    }
}
