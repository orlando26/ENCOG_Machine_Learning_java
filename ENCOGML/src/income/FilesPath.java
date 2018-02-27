package income;

public class FilesPath {
	
	private static final String BASE_PATH = FilesPath.class.getResource("/income/data/").getPath();
	
	public static final String BASE_FILE = BASE_PATH + "adult.csv";
	
	public static final String SHUFFLED_BASE_FILE = BASE_PATH + "adult_shuffled.csv";
	
	public static final String TRAINING_FILE = BASE_PATH + "adult_train.csv";
	
	public static final String EVALUATION_FILE = BASE_PATH + "adult_evaluation.csv";
	
	public static final String NORMALIZED_TRAINING_FILE = BASE_PATH + "adult_train_norm.csv";
	
	public static final String NORMALIZED_EVAL_FILE = BASE_PATH + "adult_eval_norm.csv";
	
	public static final String ANALYST_FILE = BASE_PATH + "adult_analyst.ega";
	
	public static final String TRAINED_NETWORK_FILE = BASE_PATH + "adult_network.eg";
}
