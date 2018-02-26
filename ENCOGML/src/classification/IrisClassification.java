package classification;

import java.io.File;
import java.net.URL;

import org.encog.app.analyst.csv.shuffle.ShuffleCSV;
import org.encog.util.csv.CSVFormat;

public class IrisClassification {
	public static void main(String[] args) {
		System.out.println("shuffle CSV Data file...");
		shuffle(Config.BASE_FILE);
	}
	
	public static void shuffle(String source){
		// Shuffle the CSV.
		ShuffleCSV shuffle = new ShuffleCSV();
		File csv = new File(source);
		shuffle.analyze(csv, true, CSVFormat.ENGLISH);
		shuffle.setProduceOutputHeaders(true);
		shuffle.process(new File(Config.SHUFFLED_BASE_FILE));
	}
}
