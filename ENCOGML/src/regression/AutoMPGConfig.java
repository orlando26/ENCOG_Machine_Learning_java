package regression;

public class AutoMPGConfig {
	
	private static String BASE_PATH = AutoMPGConfig.class.getResource("/regression/data/").getPath();
	
	public static final String BASE_FILE = BASE_PATH + "AutoMPG.csv";
	
	public static final String SHUFFLED_BASE_FILE = BASE_PATH + "AutoMPGShuffled.csv";
	
	public static final String TRAINING_FILE = BASE_PATH + "AutoMPGTrain.csv";
	
	public static final String EVALUATION_FILE = BASE_PATH + "AutoMPGEval.csv";
	
	public static final String NORMALIZED_TRAINING_FILE = BASE_PATH + "AutoMPGTrainNorm.csv";
	
	public static final String NORMALIZED_EVAL_FILE = BASE_PATH + "AutoMPGEvalNorm.csv";
	
	public static final String ANALYST_FILE = BASE_PATH + "AutoMPGAnalyst.ega";
	
	public static final String TRAINED_NETWORK_FILE = BASE_PATH + "AutoMPGTrain.eg";
	
	public static final String VALIDATION_RESULT = BASE_PATH + "AutoMPGValidation.csv";
}
