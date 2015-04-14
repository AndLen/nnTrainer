import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;

import java.io.*;
import java.util.*;

/**
 * Created by Andrew on 28/03/2015.
 */
public class NeuralNetworkTrainer {
    public static final int OUTPUT_NODES = 2;
    public static final double TRAINING_PROPORTION = 0.7;
    private static final String DIR = "C:\\Users\\Andrew\\Documents\\Uni 2015\\NWEN 404\\csvs\\";
    private static final String LEVEL_NAME = "level2";
    public static final String CSV = DIR + LEVEL_NAME + ".csv";
    public static String FILENAME = DIR + LEVEL_NAME + "Best";
    public static String META_FILENAME = LEVEL_NAME + "BestMeta";

    private static int nextBSSID = 0;
    private Map<String, Integer> bssidToInputNodeIndex;

    public NeuralNetworkTrainer(List<ClassifiedSample> trainingSample) {
        //Not efficient but who cares, this isn't the slow part of training!
        bssidToInputNodeIndex = assignAllBSSIDsToIDs(trainingSample);
        int numBSSIDs = bssidToInputNodeIndex.size();
        final BasicNetwork BASE_NETWORK = new BasicNetwork();
        BASE_NETWORK.addLayer(new BasicLayer(new ActivationSigmoid(), false, numBSSIDs));
        int hiddenNodes = (int) ((numBSSIDs + 2) * (2 / 3.0));
        BASE_NETWORK.addLayer(new BasicLayer(new ActivationSigmoid(), true, hiddenNodes));
        BASE_NETWORK.addLayer(new BasicLayer(new ActivationLinear(), true, OUTPUT_NODES));
        BASE_NETWORK.getStructure().finalizeStructure();
        MLDataSet[] dataSets = getDataSets(trainingSample);
        MLDataSet trainingSet = dataSets[0];
        MLDataSet validationSet = dataSets[1];
        BasicNetwork bestOverallNetwork = null;
        double bestOverallError = Double.MAX_VALUE;
        long samples = trainingSet.getRecordCount();
        double totalError = 0;
        System.out.printf("Training on %d samples\n", samples);
        for (int i = 0; i < 100; i++) {
            System.out.println("Iteration: " + i);
            BASE_NETWORK.reset();
            BasicNetwork bestResult = train(BASE_NETWORK, trainingSet, validationSet);
            double error = findOverallError(bestResult, validationSet, trainingSet);
            totalError += error;
            if (error < bestOverallError) {
                bestOverallNetwork = (BasicNetwork) bestResult.clone();
                bestOverallError = error;
            }
        }
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Avg error was: %f\n", totalError / 100));
        sb.append(String.format("Best overall network had  error of: %f/%f\n", bestOverallError, findOverallError(bestOverallNetwork, validationSet, trainingSet)));
        sb.append("Val Set Performance:\n");
        compareOutputs(validationSet, bestOverallNetwork,sb);
        sb.append("Training Set Performance:\n");
        compareOutputs(trainingSet, bestOverallNetwork,sb);
        System.out.print(sb.toString());
        EncogDirectoryPersistence.saveObject(new File(FILENAME + bestOverallError + ".nn"), bestOverallNetwork);
        writeObjectToFile(bssidToInputNodeIndex, META_FILENAME + bestOverallError + ".dat");
        writeObjectToFile(sb.toString(),META_FILENAME + bestOverallError + ".out");

    }

    public static void writeObjectToFile(Object object, String filename) {
        try {
            File dir = new File(DIR);
            dir.mkdirs();
            File file = new File(dir, filename);
            FileOutputStream out = new FileOutputStream(file, false);
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(out);
            objectOutputStream.writeObject(object);
            objectOutputStream.close();
        } catch (IOException e) {
            System.err.println("File write failed: " + e.toString());
        }
    }

    public static void main(String[] args) {
        String samples = readFromFile(CSV);
        List<ClassifiedSample> classifiedSamples = new ArrayList<>();
        for (String line : samples.split("\n")) {
            String[] vals = line.split(",");
            Double x = Double.parseDouble(vals[0]);
            Double y = Double.parseDouble(vals[1]);
            String bssid = vals[2];
            Double frequency = Double.parseDouble(vals[3]);
            List<Double> rssiReadings = new ArrayList<>();
            for (int i = 4; i < vals.length; i++) {
                rssiReadings.add(Double.parseDouble(vals[i]));
            }
            classifiedSamples.add(new ClassifiedSample(new ClassifiedSample.SampleLocation(x, y), bssid, frequency, rssiReadings));
        }
        new NeuralNetworkTrainer(classifiedSamples);
    }

    public static String readFromFile(String fileName) {

        String ret = "";

        try {
            InputStream inputStream = new FileInputStream(new File(fileName));

            InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            String receiveString = "";
            StringBuilder stringBuilder = new StringBuilder();

            while ((receiveString = bufferedReader.readLine()) != null) {
                stringBuilder.append(receiveString).append("\n");
            }

            inputStream.close();
            ret = stringBuilder.toString();
        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + e.toString());
        } catch (IOException e) {
            System.err.println("Can not read file: " + e.toString());
        }

        return ret;
    }

    /**
     * Minimum RSSI is typically around -90. Assume -100 absolute minimum. Map -100 to 0. Then
     * more positive values = closer to the location (better signal).
     * <p/>
     * The benefit of this is that 0 (no signal for that BSSID) is the worst possible result.
     * Otherwise (if we left the vals as -ve), 0 would be PERFECT signal and we'd have to have
     * "no signal" represented as -100 or something. This is best avoided because it still indicates
     * some sort of signal was found ---> the NN might just check for the existance of ANY signal
     * for some BSSIDS (e.g. where only one AP is nearby).
     * <p/>
     * Plus normalised like this is easier to reason about as it's more intuitive...
     */
    private static double normaliseRSSI(Double averageRSSI) {
        return 100 + averageRSSI;
    }

    /**
     * @param samples
     * @param bssidToInputNode Fully populated map.
     * @return
     */
    private static double[] convertSamplesToInputVector(List<? extends Sample> samples, Map<String, Integer> bssidToInputNode) {
        double[] inputs = new double[bssidToInputNode.size()];
        //Assign ones we know. REST WILL BE ZERO.
        for (Sample sample : samples) {
            Integer idForBssid = getIDForBssid(sample.getBssid(), bssidToInputNode, false);
            if (idForBssid != null) {
                inputs[idForBssid] = normaliseRSSI(sample.getAverageRSSI());
            } else {
                System.out.println("WARNING: no ID for: " + sample.getBssid());
            }
        }
        return inputs;
    }

    private static Integer getIDForBssid(String bssid, Map<String, Integer> bssidToInputNode, boolean createIfDoesntExist) {
        Integer val = bssidToInputNode.get(bssid);
        if (val == null && createIfDoesntExist) {
            val = nextBSSID++;
            bssidToInputNode.put(bssid, val);
        }
        return val;

    }

    /**
     * @param trainingSample list of (known) samples
     * @return Map of (X,Y) to list of samples.
     */
    private static Map<ClassifiedSample.SampleLocation, List<ClassifiedSample>> assignSamplesToLocations(List<ClassifiedSample> trainingSample) {
        Map<ClassifiedSample.SampleLocation, List<ClassifiedSample>> assignedSamples = new HashMap<>();
        for (ClassifiedSample classifiedSample : trainingSample) {
            ClassifiedSample.SampleLocation sampleLocation = classifiedSample.getSampleLocation();
            List<ClassifiedSample> samples = assignedSamples.get(sampleLocation);
            if (samples == null) {
                samples = new ArrayList<>();
            }
            samples.add(classifiedSample);
            assignedSamples.put(sampleLocation, samples);
        }
        return assignedSamples;
    }

    private static double findOverallError(BasicNetwork network, MLDataSet validationSet, MLDataSet trainSet) {
        double validationError = network.calculateError(validationSet);
        double trainError = network.calculateError(trainSet);

        return (trainError * (1 - TRAINING_PROPORTION) + validationError * TRAINING_PROPORTION);
    }

    private void compareOutputs(MLDataSet dataSet, BasicNetwork bestOverallNetwork, StringBuilder sb) {
        sb.append("Had error of: ").append(bestOverallNetwork.calculateError(dataSet)).append("\n");
        double totalDistanceOff = 0;
        long recordCount = dataSet.getRecordCount();
        for (int i = 0; i < recordCount; i++) {
            MLData input = dataSet.get(i).getInput();
            double[] output = bestOverallNetwork.compute(input).getData();
            double[] desiredOutput = dataSet.get(i).getIdeal().getData();
            double xOut = output[0];
            double xDesired = desiredOutput[0];
            double yOut = output[1];
            double yDesired = desiredOutput[1];
            double xDiff = Math.abs(xOut - xDesired);
            double yDiff = Math.abs(yOut - yDesired);
            double euclDistOff = Math.sqrt(xDiff*xDiff + yDiff*yDiff);
            totalDistanceOff+=euclDistOff;
            sb.append(String.format("produced outputs [%.2f,%.2f], desired outputs were [%.0f,%.0f] off-by: %.2fm\n", xOut, yOut, xDesired, yDesired, euclDistOff));
        }
        sb.append("Avg off across outputs of: ").append(totalDistanceOff/ recordCount).append("\n");
    }

    private BasicNetwork train(BasicNetwork network, MLDataSet trainingSet, MLDataSet validationSet) {

        final Propagation train = new ResilientPropagation(network, trainingSet);

        int epoch = 1;
        double bestOverallError = Double.MAX_VALUE;
        BasicNetwork bestOverallNetwork = null;
        network.clone();
        do {
            train.iteration();
            double overallError = findOverallError(network, validationSet, trainingSet);
            //  System.out.println("Epoch #" + epoch + " Training Error:" + train.getError() + " Validation Error:" + validationError);
            epoch++;
            if (overallError < bestOverallError) {
                bestOverallError = overallError;
                bestOverallNetwork = (BasicNetwork) network.clone();
            }
        } while (epoch < 500);
        train.finishTraining();
        System.out.printf("Best Overall error is %f\n", bestOverallError);

        return bestOverallNetwork;
    }

    /**
     * @param trainingSample list of (known) samples
     * @return training and validation sets (respectively)
     */
    private BasicMLDataSet[] getDataSets(List<ClassifiedSample> trainingSample) {

        //Get <X,Y> to <X,Y,rssi...>
        Map<ClassifiedSample.SampleLocation, List<ClassifiedSample>> locationSamples = assignSamplesToLocations(trainingSample);
        int numSamples = locationSamples.size();
        System.out.printf("Training set has %d vectors\n", numSamples);
        int numBSSIDs = bssidToInputNodeIndex.size();
        List<double[]> inputs = new ArrayList<>(numSamples);
        List<double[]> desiredOutputs = new ArrayList<>(numSamples);

        for (Map.Entry<ClassifiedSample.SampleLocation, List<ClassifiedSample>> entry : locationSamples.entrySet()) {
            ClassifiedSample.SampleLocation loc = entry.getKey();
            double[] desiredOutput = new double[]{loc.getX(), loc.getY()};
            //Convert this <X,Y> to readings list into an input vector
            double[] input = convertSamplesToInputVector(entry.getValue(), bssidToInputNodeIndex);
            inputs.add(input);
            desiredOutputs.add(desiredOutput);

        }
        List<Integer> indexes = new ArrayList<>(numSamples);
        for (int i = 0; i < numSamples; i++) {
            indexes.add(i);
        }
        Random rnd = new Random();
        long seed = rnd.nextLong();
        System.out.printf("Random seed: %d\n",seed);
        Collections.shuffle(indexes, new Random(seed));

        int numTraining = (int) (numSamples * TRAINING_PROPORTION);
        double[][] trainingInputs = new double[numTraining][];
        double[][] trainingOutputs = new double[numTraining][];

        for (int i = 0; i < numTraining; i++) {
            trainingInputs[i] = inputs.get(indexes.get(i));
            trainingOutputs[i] = desiredOutputs.get(indexes.get(i));
        }

        int numValidation = numSamples - numTraining;
        double[][] validationInputs = new double[numValidation][];
        double[][] validationOutputs = new double[numValidation][];

        for (int i = 0; i < numValidation; i++) {
            validationInputs[i] = inputs.get(indexes.get(i + numTraining));
            validationOutputs[i] = desiredOutputs.get(indexes.get(i + numTraining));
        }
        System.out.printf("%d total samples, %d/%d in training, %d/%d in validation\n", numSamples, trainingInputs.length, trainingOutputs.length, validationInputs.length, validationOutputs.length);
        return new BasicMLDataSet[]{new BasicMLDataSet(trainingInputs, trainingOutputs), new BasicMLDataSet(validationInputs, validationOutputs)};
    }


    private Map<String, Integer> assignAllBSSIDsToIDs(List<ClassifiedSample> samples) {
        Map<String, Integer> bssidToInputNodeIndex = new HashMap<>();
        for (ClassifiedSample sample : samples) {
            //Populates the ID <-> index map
            getIDForBssid(sample.getBssid(), bssidToInputNodeIndex,true);
        }
        return bssidToInputNodeIndex;


    }

}
