package io;

import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.criteria.EntropyLDiversity;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.metric.Metric;

public class Config {
    public static ARXConfiguration getConfig(int k){
        ARXConfiguration config = ARXConfiguration.create();
        config.addPrivacyModel(new KAnonymity(k));
        config.setSuppressionLimit(1d);
        config.setQualityModel(Metric.createLossMetric(0.5, Metric.AggregateFunction.ARITHMETIC_MEAN));
        return config;
    }
//
//    public static ARXConfiguration getConfig(int k, double l, String target){
//        ARXConfiguration config = ARXConfiguration.create();
//        config.addPrivacyModel(new KAnonymity(k));
//        config.addPrivacyModel(new EntropyLDiversity(target, l));
//        config.setSuppressionLimit(1d);
//        config.setQualityModel(Metric.createLossMetric(0.5, Metric.AggregateFunction.ARITHMETIC_MEAN));
//        return config;
//    }

    public static ARXConfiguration getConfig(int k, Metric qualityMetric){
        ARXConfiguration config = ARXConfiguration.create();
        config.addPrivacyModel(new KAnonymity(k));
        config.setSuppressionLimit(1d);
        config.setQualityModel(qualityMetric);
        return config;
    }

    public static ARXConfiguration getConfig(int k, double l, String target, Metric qualityMetric){
        ARXConfiguration config = ARXConfiguration.create();
        config.addPrivacyModel(new KAnonymity(k));
//        config.addPrivacyModel(new EntropyLDiversity("PINCP", l));
//        config.addPrivacyModel(new EntropyLDiversity("RELP", l));
        config.addPrivacyModel(new EntropyLDiversity(target, l));
        config.setSuppressionLimit(1d);
        config.setQualityModel(qualityMetric);
        return config;
    }

}
