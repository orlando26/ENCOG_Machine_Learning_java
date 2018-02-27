package regression;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

import org.encog.app.analyst.AnalystFileFormat;
import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.csv.normalize.AnalystNormalizeCSV;
import org.encog.app.analyst.csv.segregate.SegregateCSV;
import org.encog.app.analyst.csv.segregate.SegregateTargetPercent;
import org.encog.app.analyst.csv.shuffle.ShuffleCSV;
import org.encog.app.analyst.wizard.AnalystWizard;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

public class AutoMPG {
	public static void main(String[] args) {

		System.out.println("Shuffle file...");
		shuffle(AutoMPGConfig.BASE_FILE);

		System.out.println("Segregate file into training and evaluation files...");
		segregate(AutoMPGConfig.SHUFFLED_BASE_FILE);

		System.out.println("Normalize data...");
		normalize();

		System.out.println("Create the Neural Network...");
		createNetwork(AutoMPGConfig.TRAINED_NETWORK_FILE);

		System.out.println("Train neural network...");
		trainNetwork();
		
//		System.out.println("Evaluation...");
//		evaluate();
	}

	public static void shuffle(String source){
		// Shuffle the base file
		ShuffleCSV shuffle = new ShuffleCSV();
		shuffle.analyze(new File(source), true, CSVFormat.ENGLISH);
		shuffle.setProduceOutputHeaders(true);
		shuffle.process(new File(AutoMPGConfig.SHUFFLED_BASE_FILE));
	}

	public static void segregate(String source){
		SegregateCSV seg = new SegregateCSV();
		seg.getTargets().add(new SegregateTargetPercent(new File(AutoMPGConfig.TRAINING_FILE), 75));
		seg.getTargets().add(new SegregateTargetPercent(new File(AutoMPGConfig.EVALUATION_FILE), 25));
		seg.setProduceOutputHeaders(true);
		seg.analyze(new File(source), true, CSVFormat.ENGLISH);
		seg.process();
	}

	public static void normalize(){
		// Analyst
		EncogAnalyst analyst = new EncogAnalyst();

		// Wizard
		AnalystWizard wizard = new AnalystWizard(analyst);
		wizard.wizard(new File(AutoMPGConfig.BASE_FILE), true, AnalystFileFormat.DECPNT_COMMA);

		// Cylinders
		analyst.getScript().getNormalize().getNormalizedFields().get(0).setAction(NormalizationAction.Equilateral);
		// Displacement
		analyst.getScript().getNormalize().getNormalizedFields().get(1).setAction(NormalizationAction.Normalize);
		// HorsePower
		analyst.getScript().getNormalize().getNormalizedFields().get(2).setAction(NormalizationAction.Normalize);
		// Weight
		analyst.getScript().getNormalize().getNormalizedFields().get(3).setAction(NormalizationAction.Normalize);
		// Acceleration
		analyst.getScript().getNormalize().getNormalizedFields().get(4).setAction(NormalizationAction.Normalize);
		// Year
		analyst.getScript().getNormalize().getNormalizedFields().get(5).setAction(NormalizationAction.Equilateral);
		// Origin
		analyst.getScript().getNormalize().getNormalizedFields().get(6).setAction(NormalizationAction.Equilateral);
		// Name
		analyst.getScript().getNormalize().getNormalizedFields().get(7).setAction(NormalizationAction.Ignore);
		// Mpg
		analyst.getScript().getNormalize().getNormalizedFields().get(8).setAction(NormalizationAction.Normalize);


		//Norm for training
		AnalystNormalizeCSV norm = new AnalystNormalizeCSV();
		norm.setProduceOutputHeaders(true);

		norm.analyze(new File(AutoMPGConfig.TRAINING_FILE), true, CSVFormat.ENGLISH, analyst);
		norm.normalize(new File(AutoMPGConfig.NORMALIZED_TRAINING_FILE));

		//Norm for evaluation
		norm.analyze(new File(AutoMPGConfig.EVALUATION_FILE), true, CSVFormat.ENGLISH, analyst);
		norm.normalize(new File(AutoMPGConfig.NORMALIZED_EVAL_FILE));

		// Save the analyst file
		analyst.save(new File(AutoMPGConfig.ANALYST_FILE));
	}

	public static void createNetwork(String networkFile){
		BasicNetwork network = new BasicNetwork();

		network.addLayer(new BasicLayer(new ActivationLinear(), true, 22));
		network.addLayer(new BasicLayer(new ActivationTANH(), true, 6));
		network.addLayer(new BasicLayer(new ActivationTANH(), true, 1));
		network.getStructure().finalizeStructure();
		network.reset();
		EncogDirectoryPersistence.saveObject(new File(networkFile), network);
	}

	public static void trainNetwork(){
		BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File(AutoMPGConfig.TRAINED_NETWORK_FILE));
		MLDataSet trainingSet = EncogUtility.loadCSV2Memory(AutoMPGConfig.NORMALIZED_TRAINING_FILE, 
				network.getInputCount(), network.getOutputCount(), true, CSVFormat.ENGLISH, false);

		ResilientPropagation train = new ResilientPropagation(network, trainingSet);

		int epoch = 1;
		do{
			train.iteration();
			System.out.println("epoch: " + epoch + " error: " + train.getError());
			epoch++;
		}while(train.getError() > 0.01);
		EncogDirectoryPersistence.saveObject(new File(AutoMPGConfig.TRAINED_NETWORK_FILE), network);
	}

	public static void evaluate(){
		BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.loadObject(new File(AutoMPGConfig.TRAINED_NETWORK_FILE));
		EncogAnalyst analyst = new EncogAnalyst();
		analyst.load(AutoMPGConfig.ANALYST_FILE);
		MLDataSet evaluationSet = EncogUtility.loadCSV2Memory(AutoMPGConfig.NORMALIZED_EVAL_FILE, 
				network.getInputCount(), network.getOutputCount(), true, CSVFormat.ENGLISH, false);

		try (PrintWriter out = new PrintWriter(AutoMPGConfig.VALIDATION_RESULT)) {

			for (MLDataPair item : evaluationSet) {
				BasicMLData normalizeActualOutput = (BasicMLData) network.compute(item.getInput());
				double actualOutput = analyst.getScript().getNormalize().getNormalizedFields().get(8).deNormalize(normalizeActualOutput.getData(0));
				double idealOutput = analyst.getScript().getNormalize().getNormalizedFields().get(8).deNormalize(item.getIdeal().getData(0));
				
				out.println(idealOutput + "," + actualOutput);
				System.out.println(idealOutput + ", " + actualOutput);
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}
