package classification;

import java.net.URL;

public class Config {
	private static String basePath = "/classification/data/";
	
	public static final String BASE_FILE = 
			Config.class.getResource(basePath + "irisData.csv").getPath();
	
	public static final String SHUFFLED_BASE_FILE = 
			Config.class.getResource(basePath).getPath() + "iris_shuffled.csv";
}
