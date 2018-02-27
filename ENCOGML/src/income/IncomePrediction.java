package income;

import java.io.File;

import org.encog.app.analyst.AnalystFileFormat;
import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.csv.normalize.AnalystNormalizeCSV;
import org.encog.app.analyst.csv.segregate.SegregateCSV;
import org.encog.app.analyst.csv.segregate.SegregateTargetPercent;
import org.encog.app.analyst.csv.shuffle.ShuffleCSV;
import org.encog.app.analyst.wizard.AnalystWizard;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;




public class IncomePrediction {

	public static void main(String[] args) {
		System.out.println("Shuffle file...");
		shuffle(FilesPath.BASE_FILE);

		System.out.println("Segregate file into training and evaluation...");
		segregate(FilesPath.SHUFFLED_BASE_FILE);

		System.out.println("Normalize data...");
		normalize();
		
		System.out.println("Create network...");
		createNetwork(FilesPath.TRAINED_NETWORK_FILE);
		
		System.out.println("Training network...");
		trainNetwork();
	}

	public static void shuffle(String source){
		// Shuffle the CSV.
		ShuffleCSV shuffle = new ShuffleCSV();
		File csv = new File(source);
		shuffle.analyze(csv, true, CSVFormat.ENGLISH);
		shuffle.setProduceOutputHeaders(true);
		shuffle.process(new File(FilesPath.SHUFFLED_BASE_FILE));
	}

	public static void segregate(String source){
		// Segregate file into training and evaluation files
		SegregateCSV seg = new SegregateCSV();

		seg.getTargets().add(new SegregateTargetPercent(new File(FilesPath.TRAINING_FILE), 75));
		seg.getTargets().add(new SegregateTargetPercent(new File(FilesPath.EVALUATION_FILE), 25));
		seg.setProduceOutputHeaders(true);
		seg.analyze(new File(source), true, CSVFormat.ENGLISH);
		seg.process();
	}

	public static void normalize(){
		// Analyst
		EncogAnalyst analyst = new EncogAnalyst();

		// Wizard
		AnalystWizard wizard = new AnalystWizard(analyst);
		wizard.wizard(new File(FilesPath.BASE_FILE), true, AnalystFileFormat.DECPNT_COMMA);


		// Norm for Training
		AnalystNormalizeCSV norm = new AnalystNormalizeCSV();
		norm.analyze(new File(FilesPath.TRAINING_FILE), true, CSVFormat.ENGLISH, analyst);
		norm.setProduceOutputHeaders(true);
		norm.normalize(new File(FilesPath.NORMALIZED_TRAINING_FILE));

		// Norm for evaluation
		norm.analyze(new File(FilesPath.EVALUATION_FILE), true, CSVFormat.ENGLISH, analyst);
		norm.normalize(new File(FilesPath.NORMALIZED_EVAL_FILE));

		// Save the Analyst file
		analyst.save(new File(FilesPath.ANALYST_FILE));	
	}

	public static void createNetwork(String networkFile){
		BasicNetwork network = new BasicNetwork();

		network.addLayer(new BasicLayer(new ActivationLinear(), true, 14));
		network.addLayer(new BasicLayer(new ActivationTANH(), true, 6));
		network.addLayer(new BasicLayer(new ActivationTANH(), true, 2));
		network.getStructure().finalizeStructure();
		network.reset();
		EncogDirectoryPersistence.saveObject(new File(networkFile), network);
	}

	public static void trainNetwork(){
		BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.loadObject(new File(FilesPath.TRAINED_NETWORK_FILE));
		MLDataSet trainingSet = EncogUtility.loadCSV2Memory(FilesPath.NORMALIZED_TRAINING_FILE, network.getInputCount(),
				network.getOutputCount(), true, CSVFormat.ENGLISH, false);

		ResilientPropagation train = new ResilientPropagation(network, trainingSet);
		int epoch = 1;
		do{
			train.iteration();
			System.out.println("Epoch: " + epoch + ", error: " + train.getError());
			epoch++;
		}while(train.getError() > 0.038);

		EncogDirectoryPersistence.saveObject(new File(FilesPath.TRAINED_NETWORK_FILE), network);
	}
}
