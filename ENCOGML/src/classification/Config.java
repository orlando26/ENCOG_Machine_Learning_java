package classification;

import java.net.URL;

public class Config {
	private static String basePath = "/classification/data/";
	
	public static final String BASE_FILE = 
			Config.class.getResource(basePath + "irisData.csv").getPath();
	
	public static final String SHUFFLED_BASE_FILE = 
			Config.class.getResource(basePath).getPath() + "iris_shuffled.csv";
	
	public static final String TRAINING_FILE = 
			Config.class.getResource(basePath).getPath() + "iris_training.csv";
	
	public static final String EVALUATION_FILE = 
			Config.class.getResource(basePath).getPath() + "iris_eval.csv";
	
	public static final String NORMALIZED_TRAINING_FILE = 
			Config.class.getResource(basePath).getPath() + "iris_train_norm.csv";
	
	public static final String NORMALIZED_EVAL_FILE = 
			Config.class.getResource(basePath).getPath() + "iris_eval_norm.csv";
	
	public static final String ANALYST_FILE = 
			Config.class.getResource(basePath).getPath() + "iris_analyst.ega";
	
	public static final String TRAINED_NETWORK_FILE = 
			Config.class.getResource(basePath).getPath() + "iris_train.eg";
}
