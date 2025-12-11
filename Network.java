import java.io.*;
import java.util.*;
import com.google.gson.*;

/**
 * Author: Mihir Kotbagi
 * Date of creation: 9/5/25
 * Date of most recent modification: 12/10/25
 * Description: The Network class implements an N-layer neural network that uses backpropagation for training. It is written against
 * the "N-Layer" design document. Configuration parameters are read from a user-selected file using Google's GSON library for JSON
 * parsing.
 * 
 * Table of Contents:
 * void setConfigurationParameters()
 * void loadConfigurationParameters() throws FileNotFoundException, IOException
 * void echoConfigurationParameters()
 * void allocateMemory()
 * void populateArrays()
 * void randomPopulateWeigts()
 * void loadWeights() throws FileNotFoundException, IOException
 * void saveWeights() throws FileNotFoundException, IOException
 * void manualPopulateTruth()
 * void loadTruth() throws FileNotFoundException, IOException
 * void manualPopulateTests()
 * void loadTests() throws FileNotFoundException, IOException
 * double randomize(double low, double high)
 * double linear(double x)
 * double linearDerivative(double x)
 * double sigmoid(double x)
 * double sigmoid(double x)
 * double sigmoidDerivative(double x)
 * double tanh(double x)
 * double tanhDerivative(double x)
 * double activationFunction(double x)
 * double activationFunctionDerivative(double x)
 * void populateInputActivations(int testCase)
 * void runOrTrain() throws FileNotFoundException, IOException
 * void run(int testCase)
 * double runForTrain(int testCase)
 * void gradientDescent(int testCase)
 * void reportResults()
 * public static void main(String[] args)
 */
public class Network 
{   
   final static int RAND = 1;               // Set weightPop to RAND to randomly populate the weight array
   final static int FILE_LOAD = 2;          // Set weightPop to FILE_LOAD to populate the weight array from a file 
   final static double NANOSECONDS_PER_SECOND = 1e9;

   int layers;                // Number of connectivity layers in the network   
   int activationLayers;      // Number of activation layers in the network, one more than LAYERS
   
   int A_INDEX = 0;           // n-index of the input activation layer
   int H1_INDEX = 1;          // n-index of the first hidden activation layer
   int F_INDEX;               // n-index of the output activation layer
   
   int[] layerSizes;          // Configuration array that stores number of nodes in each layer (input, hidden 1, hidden 2, output)

   double[][] a;              // 2D array for all network activations

   double[][][] weights;

   double lambda;             // Learning factor

   int numTestCases;          // Number of test cases

   double[][] testCases;      // 2D array storing test inputs
   double[][] truthValues;    // 2D array storing truth values for the test cases

   String test;               // "AND", "OR", "XOR", or "AND_OR_XOR" depending on what the user wants to run/train against

   double maxErrorThreshold;  // Maximum acceptable average error across all test cases while training

   int weightPop;             // As described above, weightPop is set to 1 or 2 depending on population method
   double randLow;            // Minimum value for randomization function
   double randHigh;           // Maximum value for randomization function
   String weightLoadPath;     // Path to file weights should be loaded from
   String weightSavePath;     // Path to file weights should be saved to

   String configLoadPath;     // Path to file configuration parameters should be loaded from
   String testCaseLoadPath;   // Path to file containing test cases (used if loadTests is true)
   String truthTableLoadPath; // Path to file containig truth table (used if loadTruth is true)

   boolean train;             // True if the network should be trained, false if the network should only be run
   boolean printTruth;        // True if the user wants to output the truth table, false otherwise
   boolean printWeights;      // True if the user wants to output the weight arrays, false otherwise
   boolean saveWeights;       // True if the user wants to save the weights, false otherwise
   boolean loadTests;         // True if the user wants to load the test cases from a file, false otherwise
   boolean loadTruth;         // True if the user wants to load the truth table from a file, false otherwise

   int maxIterations;         // Maximum number of iterations before network quits training
   int weightSaveFreq;        // How many training iterations to wait between saving weights
   int keepAliveFreq;         // How many training iterations to wait between printing the keep alive status message
   
   int iterations;            // Number of iterations used for training
   double averageError;       // Average error from the previous training iteration

   double[][] Theta;
   double[][] psi;
   
   double[][] outputs;        // Stores outputs for each test case to report (distinct from output layer of activations)

   long runOrTrainStart;      // Used to time running/training, in nanoseconds
   long runOrTrainFinish;     // Used to time running/training, in nanoseconds

/**
 * Allows the user to set parameters that configure the network and decide between training and running. Also allows the user
 * to specify if they want to print supplementary information (truth table and weights).
 */
   void setConfigurationParameters() 
   {
      activationLayers = 4;
      layers = 3;
      
      layerSizes = new int[activationLayers];

      A_INDEX = 0;
      H1_INDEX = 1;
      F_INDEX = 3;
      int H2_INDEX = 2;

      layerSizes[A_INDEX] = 2;
      layerSizes[H1_INDEX] = 5;
      layerSizes[H2_INDEX] = 5;
      layerSizes[F_INDEX] = 3;

      train = true;

      numTestCases = 4;

      test = "AND_OR_XOR";

      lambda = 0.3;
      maxErrorThreshold = 0.0002;
      maxIterations = 100000;
      weightSaveFreq = 5000;

      weightPop = RAND;
      weightLoadPath = "weightLoad.bin";

      randLow = 0.1;
      randHigh = 1.5;

      printTruth = true;
      printWeights = false;

      saveWeights = true;
      weightSavePath = "weightSave.bin";

      loadTests = true;
      loadTruth = true;

      testCaseLoadPath = "AND_OR_XOR_test.txt";
      truthTableLoadPath = "AND_OR_XOR_truth.txt";
   } // void setConfigurationParameters()

/**
 * Loads configuration parameters from a control file specified by the user
 */
   void loadConfigurationParameters() throws FileNotFoundException, IOException
   {
      FileReader reader = new FileReader(configLoadPath);

      JsonObject parameters = JsonParser.parseReader(reader).getAsJsonObject();

      activationLayers = parameters.get("activationLayers").getAsInt();
      layers = activationLayers - 1; // Number of connectivity layers is always one less than number of activation layers
      F_INDEX = layers;              // Output index equals number of connectivity layers

      JsonArray layerSizesJSON = parameters.get("layerSizes").getAsJsonArray();

      layerSizes = new int[activationLayers];

      for (int n = A_INDEX; n < activationLayers; n++)
      {
         layerSizes[n] = layerSizesJSON.get(n).getAsInt();
      }

      train = parameters.get("train").getAsBoolean();

      numTestCases = parameters.get("numTestCases").getAsInt();

      test = parameters.get("test").getAsString();
      testCaseLoadPath = parameters.get("testCasePath").getAsString();

      truthTableLoadPath = parameters.get("truthTablePath").getAsString();

      lambda = parameters.get("lambda").getAsDouble();
      maxErrorThreshold = parameters.get("maxErrorThreshold").getAsDouble();
      maxIterations = parameters.get("maxIterations").getAsInt();

      String weightPopName = parameters.get("weightPop").getAsString();
      switch (weightPopName)
      {
         case "RAND":
            weightPop = RAND;
            break;
         case "FILE_LOAD":
            weightPop = FILE_LOAD;
            break;
         default:
            System.out.println("Invalid population method specified. Populating weights randomly.");
            weightPop = RAND;
      } // switch (weightPopName)

      weightLoadPath = parameters.get("weightLoadPath").getAsString();

      randLow = parameters.get("randLow").getAsDouble();
      randHigh = parameters.get("randHigh").getAsDouble();

      printTruth = parameters.get("printTruth").getAsBoolean();
      printWeights = parameters.get("printWeights").getAsBoolean();
      saveWeights = parameters.get("saveWeights").getAsBoolean();

      weightSavePath = parameters.get("weightSavePath").getAsString();

      loadTests = parameters.get("loadTests").getAsBoolean();
      loadTruth = parameters.get("loadTruth").getAsBoolean();

      weightSaveFreq = parameters.get("weightSaveFreq").getAsInt();
      keepAliveFreq = parameters.get("keepAliveFreq").getAsInt();

      if(keepAliveFreq <= 0)
      {
         keepAliveFreq = 0;
      }
   } // void loadConfigurationParameters() throws FileNotFoundException, IOException
   
/**
 * Prints out all the configuration parameters specified by the user.
 */
   void echoConfigurationParameters() 
   {
      System.out.println("=====================================================================");
      System.out.println("The configuration parameters were loaded from " + configLoadPath + ".");
      System.out.print("Network configuration: ");
      for (int n = A_INDEX; n < activationLayers; n++)
      {
         System.out.print(layerSizes[n]);

         if (n < activationLayers - 1) // Hyphens are used for separation when is at least one value yet to be printed
         {
            System.out.print("-");
         } else {
            System.out.print(".");
         }
      } // for (int n = A_INDEX; n < activationLayers; n++)
      System.out.println();
      
      switch (weightPop) 
      {
         case RAND:
            System.out.println("The weights will be populated randomly, and the range for random weight population is ["
                  + randLow + ", " + randHigh + "].");
            break;
         case FILE_LOAD:
            System.out.println("The weights will be loaded from " + weightSavePath + ".");
            break;
      } // switch (weightPop)

      if (train) 
      {
         System.out.println("Network is training against " + test + ". \n\nTraining parameters:");
         System.out.println("The maximum number of iterations is " + maxIterations + ".");
         System.out.println("Weights will be saved every " + weightSaveFreq + " iterations.");
         if (keepAliveFreq != 0)
         {
            System.out.println("A keep alive message will be printed every " + keepAliveFreq + " iterations.");
         }
         System.out.println("The maximum error threshold is " + maxErrorThreshold + ".");
         System.out.println("The learning factor is " + lambda + ".");
      } // if (train)
      else 
      {
         System.out.println("Network is running against " + test + ".");
      }

      if (loadTests)
      {
         System.out.println("The test cases were loaded from " + testCaseLoadPath + ".");
      }
      else
      {
         System.out.println("The test cases were populated using values defined in the manualPopulateTests() function.");
      }

      if (train || printTruth)
      {
         if (loadTruth)
         {
            System.out.println("The truth table was loaded from " + truthTableLoadPath + ".");
         }
         else
         {
            System.out.println("The truth table was populated using values defined in the manualPopulateTruth() function.");
         }
      } // if (train || printTruth)

      System.out.println("=====================================================================");
   } // void echoConfigurationParameters()

/**
 * Uses the configuration parameters to declare all the arrays used. If not training the network, only arrays needed for
 * running are declared.
 */
   void allocateMemory() 
   {
      a = new double[activationLayers][]; // The total number of activation layers equals one more than the number of layers
      
      for (int n = A_INDEX; n < activationLayers; n++)
      {
         a[n] = new double[layerSizes[n]];
      }
      
      weights = new double[layers][][];
      
      for (int n = A_INDEX; n < layers; n++)
      {
         weights[n] = new double[layerSizes[n]][layerSizes[n + 1]];
      }

      testCases = new double[numTestCases][layerSizes[A_INDEX]];

      outputs = new double[numTestCases][layerSizes[F_INDEX]];

      if (train || printTruth) // The truth table is only needed if the user is training or wants to print it
      {
         truthValues = new double[numTestCases][layerSizes[F_INDEX]];  
      }

      if (train) 
      {
         Theta = new double[layers][]; // Theta and psi are 1-indexed to align with the activation layer indices
         
         for (int n = A_INDEX; n < layers; n++)
         {
            Theta[n] = new double[layerSizes[n]];
         }

         psi = new double[activationLayers][];
         
         for (int n = A_INDEX; n < activationLayers; n++)
         {
            psi[n] = new double[layerSizes[n]];
         }
      } // if (train)
   } // void allocateMemory()

/**
 * Populates the truth table and weight arrays according to the user specification. The truth table isn't populated if the user
 * isn't training and doesn't want to view it; the tests are always populated because they are used for training and running.
 */
   void populateArrays() throws FileNotFoundException, IOException
   {
      if (weightPop == RAND) 
      {
         randomPopulateWeights();
      }
      else if (weightPop == FILE_LOAD)
      {
         loadWeights();
      }

      if (train || printTruth)
      {
         if (loadTruth)
         {
            loadTruth();
         }
         else
         {
            manualPopulateTruth();
         }
      } // if (train || printTruth)
      
      if (loadTests)
      {
         loadTests();
      }
      else
      {
         manualPopulateTests();
      }
   } // void populateArrays() throws FileNotFoundException, IOException
   
/**
 * Populates the weight arrays randomly; each weight is set to a random double precision value between randLow and randHigh
 */
   void randomPopulateWeights() 
   {
      for (int n = A_INDEX; n < layers; n++)
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            for (int k = 0; k < layerSizes[n + 1]; k++)
            {
               weights[n][j][k] = randomize(randLow, randHigh);
            }
         }
      } // for (int n = A_INDEX; n < layers; n++)
   } // void randomPopulateWeights()

/**
 * Loads the weights from a file
 */
   void loadWeights() throws FileNotFoundException, IOException
   {
      File weightFile = new File(weightLoadPath);
      DataInputStream dataIn = new DataInputStream(new FileInputStream(weightFile));

      int layerSizeValidation_n; // Used to verify that the config saved in the weight file matches the user config
                         
      for (int n = A_INDEX; n < activationLayers; n++)
      {
         layerSizeValidation_n = dataIn.readInt();

         if(layerSizeValidation_n != layerSizes[n])
         {
            dataIn.close();
            throw new IOException("Weights from " + weightLoadPath + " don't match network config, so they couldn't be loaded.");
         }
      } // for (int n = A_INDEX; n < activationLayers; n++)

      for (int n = A_INDEX; n < layers; n++)
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            for (int k = 0; k < layerSizes[n + 1]; k++)
            {
               weights[n][j][k] = dataIn.readDouble();
            }
         }
      } // for (int n = A_INDEX; n < layers; n++)
      
      dataIn.close();
   } // void loadWeights() throws FileNotFoundException, IOException

/**
 * Saves the weights (in binary) to the user-specified weight save file
 */
   void saveWeights() throws FileNotFoundException, IOException
   {
      File weightFile = new File(weightSavePath);
      DataOutputStream dataOut = new DataOutputStream(new FileOutputStream(weightFile));

      for (int n = A_INDEX; n < activationLayers; n++)
      {
         dataOut.writeInt(layerSizes[n]); // Saves network configuration, which is checked when loading weights from a file
      }

      for (int n = A_INDEX; n < layers; n++)
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            for (int k = 0; k < layerSizes[n + 1]; k++)
            {
               dataOut.writeDouble(weights[n][j][k]);
            }
         }
      } // for (int n = A_INDEX; n < layers; n++)

      dataOut.close();
   } // void saveWeights() throws FileNotFoundException, IOException
   
/**
 * Populates the truth table for the test problem specified by the user.
 */
   void manualPopulateTruth() 
   {
      switch (test) 
      {
         case "AND_OR_XOR":
            truthValues[0][0] = 0.0;
            truthValues[1][0] = 0.0;
            truthValues[2][0] = 0.0;
            truthValues[3][0] = 1.0;

            truthValues[0][1] = 0.0;
            truthValues[1][1] = 1.0;
            truthValues[2][1] = 1.0;
            truthValues[3][1] = 1.0;

            truthValues[0][2] = 0.0;
            truthValues[1][2] = 1.0;
            truthValues[2][2] = 1.0;
            truthValues[3][2] = 0.0;
            break;

         case "AND":
            truthValues[0][0] = 0.0;
            truthValues[1][0] = 0.0;
            truthValues[2][0] = 0.0;
            truthValues[3][0] = 1.0;
            break;
      
         case "OR":
            truthValues[0][0] = 0.0;
            truthValues[1][0] = 1.0;
            truthValues[2][0] = 1.0;
            truthValues[3][0] = 1.0;
            break;

         case "XOR":
            truthValues[0][0] = 0.0;
            truthValues[1][0] = 1.0;
            truthValues[2][0] = 1.0;
            truthValues[3][0] = 0.0;
            break;
      } // switch (test)
   } // void populateTruth()

/**
 * Loads the truth table from a user-specified file
 */   
   void loadTruth() throws FileNotFoundException, IOException
   {
      File truthFile = new File(truthTableLoadPath);
      Scanner scanner = new Scanner(truthFile);

      for (int testCase = 0; testCase < numTestCases; testCase++)
      {
         for (int i = 0; i < layerSizes[F_INDEX]; i++) 
         {
            truthValues[testCase][i] = scanner.nextDouble();
         }
      } // for (int testCase = 0; testCase < numTestCases; testCase++)
      
      scanner.close();
   } // void loadTruth() throws FileNotFoundException, IOException

/**
 * Populates the standard test cases for binary problems
 */
   void manualPopulateTests() 
   {
      testCases[0][0] = 0.0;
      testCases[0][1] = 0.0;

      testCases[1][0] = 0.0;
      testCases[1][1] = 1.0;

      testCases[2][0] = 1.0;
      testCases[2][1] = 0.0;

      testCases[3][0] = 1.0;
      testCases[3][1] = 1.0;
   } // void manualPopulateTests() 

/**
 * Loads test cases from a user-specified file
 */   
   void loadTests() throws FileNotFoundException, IOException
   {
      File testFile = new File(testCaseLoadPath);
      Scanner scanner = new Scanner(testFile);

      for (int testCase = 0; testCase < numTestCases; testCase++)
      {
         for (int k = 0; k < layerSizes[A_INDEX]; k++) 
         {
            testCases[testCase][k] = scanner.nextDouble();
         }
      } // for (int testCase = 0; testCase < numTestCases; testCase++)
      
      scanner.close();
   } // void loadTests() throws FileNotFoundException, IOException

/**
 * Returns a random double precision floating point number between low and high.
 * Low must be less than or equal to high.
 */
   double randomize(double low, double high) 
   {
      return (high - low) * Math.random() + low;
   } // double randomize(double low, double high)

/**
 * Basic linear activation function f(x) = x
 */
   double linear(double x) 
   {
      return x;
   } // double linear(double x)

/**
 * Derivative of linear activation function f(x) = x; f'(x) = 1
 */
   double linearDerivative(double x)
   {
      return 1.0;
   } // double linearDerivative(double x)

/**
 * Returns the value of the sigmoid function for a given value of x
 */   
   double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } // double sigmoid(double x)

/**
 * Returns the value of the derivative of the sigmoid function for a given value of x
 */
   double sigmoidDerivative(double x)
   {
      double fX = sigmoid(x);
      return fX * (1.0 - fX);
   } // double sigmoidDerivative(double x)

/**
 * Returns the value of the hyperbolic tangent function for a given value of x
 */
   double tanh(double x)
   {
      double epsX = (x < 0) ? 1.0 : -1.0;
      double e_eps_2X = Math.exp(epsX * 2.0 * x);
      return epsX * (e_eps_2X - 1.0) / (e_eps_2X + 1.0);
   } // double tanh(double x)

/**
 * Returns the value of the derivative of the hyperbolic tangent function for a given value of x
 */
   double tanhDerivative(double x)
   {
      double fX = tanh(x);
      return 1.0 - fX * fX;
   } // double tanhDerivative(double x)

/**
 * Returns the value of the activation function
 */
   double activationFunction(double x)
   {
      return sigmoid(x);
   } // double activationFunction(double x)

/**
 * Returns the derivative of the activation function
 */
   double activationFunctionDerivative(double x)
   {
      return sigmoidDerivative(x);
   } // double activationFunctionDerivative(double x)

/**
 * Populates the nodes in the input activation layer using the input nodes from a certain test case
 */
   void populateInputActivations(int testCase)
   {
      for (int k = 0; k < layerSizes[A_INDEX]; k++)
      {
         a[A_INDEX][k] = testCases[testCase][k];
      }
   } // void populateInputActivations(int testCase)
   
/**
 * Runs or trains depending on what the user has configured. A sigmoid activation function is used to calculate the 
 * activations, and gradient descent is used to update the weights.
 */
   void runOrTrain() throws FileNotFoundException, IOException
   {
      runOrTrainStart = System.nanoTime();

      if (train)
      {
         iterations = 0;
         averageError = Double.MAX_VALUE; // The error must start above the threshold, otherwise training will not happen
         
         while (iterations < maxIterations && averageError > maxErrorThreshold) 
         {
            averageError = 0.0;
            
            for (int testCase = 0; testCase < numTestCases; testCase++) 
            {
               populateInputActivations(testCase);
               averageError += runForTrain(testCase);
               gradientDescent(testCase);
            } // for (int testCase = 0; testCase < numTestCases; testCase++)
            
            iterations++;

            if (saveWeights && iterations % weightSaveFreq == 0)
            {
               saveWeights();
            }

            if (keepAliveFreq != 0 && iterations % keepAliveFreq == 0)
            {
               System.out.printf("Iteration %d, Error = %f\n", iterations, averageError);
            }
/**
 * averageError is summed over all testcases in the above loop, so it must be divided by numTestCases to obtain the actual average
 * It is also divided by 2 here to optimize error calculation, as explained in the design documents
 */ 
            averageError /= (double) (2 * numTestCases);
         } // while (iterations < maxIterations && averageError > maxErrorThreshold)

         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            populateInputActivations(testCase);
            run(testCase);

            for (int i = 0; i < layerSizes[F_INDEX]; i++)
            {
               outputs[testCase][i] = a[F_INDEX][i];
            }
         } // for (int testCase = 0; testCase < numTestCases; testCase++)
      } // if (train)
      else
      {
         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            populateInputActivations(testCase);
            run(testCase);
            
            for (int i = 0; i < layerSizes[F_INDEX]; i++)
            {
               outputs[testCase][i] = a[F_INDEX][i];
            }
         } // for (int testCase = 0; testCase < numTestCases; testCase++)
      } // else: if (train)

      runOrTrainFinish = System.nanoTime();
   } // void runOrTrain()
   
/**
 * Runs the network for a particular test case; dot products aren't stored because they aren't needed for running without 
 * training.
 */
   void run(int testCase)
   {
      double Theta;

      for (int n = H1_INDEX; n < activationLayers; n++)
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            Theta = 0.0;

            for (int k = 0; k < layerSizes[n - 1]; k++)
            {
               Theta += a[n - 1][k] * weights[n - 1][k][j];
            }

            a[n][j] = activationFunction(Theta);
         } // for (int j = 0; j < layerSizes[n]; j++)
      } // for (int n = H1_INDEX; n < activationLayers; n++)
   } // void run(int testCase)

/**
 * Runs the network for a particular test case and stores all dot products.
 */
   double runForTrain(int testCase)
   {
      Double error = 0.0;

      for (int n = H1_INDEX; n < activationLayers - 1; n++) // The last activation layer is handled separately
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            Theta[n][j] = 0.0;

            for (int k = 0; k < layerSizes[n - 1]; k++)
            {
               Theta[n][j] += a[n - 1][k] * weights[n - 1][k][j];
            }

            a[n][j] = activationFunction(Theta[n][j]);
         } // for (int j = 0; j < layerSizes[n]; j++)
      } // for (int n = H1_INDEX; n < activationLayers - 1; n++)

      int n = F_INDEX;

      for (int j = 0; j < layerSizes[n]; j++)
      {
         double Theta_j = 0.0;
         
         for (int k = 0; k < layerSizes[n - 1]; k++)
         {
            Theta_j += a[n - 1][k] * weights[n - 1][k][j];
         }
         
         a[n][j] = activationFunction(Theta_j);

         double omega_i = truthValues[testCase][j] - a[n][j];
         psi[n][j] = omega_i * activationFunctionDerivative(Theta_j);
         error += omega_i * omega_i;
      } // for (int j = 0; j < layerSizes[n]; j++)

      return error;
   } // void runForTrain(int testCase)

/**
 * Updates the weights after calculating how much they should change
 */
   void gradientDescent(int testCase)
   {
      for (int n = F_INDEX - 1; n > H1_INDEX; n--) // Loops from the last activation layer to the second activation layer
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            double Omega_j = 0.0;

            for (int k = 0; k < layerSizes[n + 1]; k++)
            {
               Omega_j += psi[n + 1][k] * weights[n][j][k];
               weights[n][j][k] += lambda * a[n][j] * psi[n + 1][k];
            }

            psi[n][j] = Omega_j * activationFunctionDerivative(Theta[n][j]);
         } // for (int j = 0; j < layerSizes[n]; j++)
      } // for (int n = F_INDEX - 1; n > H1_INDEX; n--)

      int n = H1_INDEX; // The first activation layer has a special loop and is handled separately

      for (int j = 0; j < layerSizes[n]; j++)
      {
         double Omega_j = 0.0;

         for (int k = 0; k < layerSizes[n + 1]; k++)
         {
            Omega_j += psi[n + 1][k] * weights[n][j][k];
            weights[n][j][k] += lambda * a[n][j] * psi[n + 1][k];
         }

         psi[n][j] = Omega_j * activationFunctionDerivative(Theta[n][j]);

         for (int m = 0; m < layerSizes[n - 1]; m++)
         {
            weights[n - 1][m][j] += lambda * a[n - 1][m] * psi[n][j];
         }
      } // for (int j = 0; j < layerSizes[n]; j++)
   } // void calcGradientDescent(int testCase)
   
/**
 * Reports the results for running or training, and depending on configuration parameters, outputs test cases and weights as well.
 */
   void reportResults()
   {
      System.out.println("\n=================================================");
      System.out.println("Network attempted to " + (train ? "train" : "run") + " on " + numTestCases + " test cases.");
      System.out.println((train ? "Training" : "Running") + " took " +
         String.format("%,.4f", (runOrTrainFinish - runOrTrainStart) / NANOSECONDS_PER_SECOND) + " seconds.");

      if (train)
      {
         System.out.print("Training exited because ");

         if (iterations >= maxIterations && averageError <= maxErrorThreshold)
         {
            System.out.print("the number of iterations exceeded maxIterations and the error threshold was reached.");
         }
         else if (iterations >= maxIterations)
         {
            System.out.print("the number of iterations exceeded maxIterations.");
         }
         else if (averageError <= maxErrorThreshold)
         {
            System.out.print("the error threshold was reached.");
         }

         System.out.println("\nAverage Error: " + String.format("%,.4f",averageError));
         System.out.println("Iterations: " + iterations);

         if (saveWeights)
         {
            System.out.println("Weights saved to " + weightSavePath);
         }
      } // if (train)

      System.out.println("\nResults:");
      if (printTruth)
      {
         for (int k = 0; k < layerSizes[A_INDEX]; k++)
         {
            System.out.print("a" + k + "   |");
         }
         System.out.println("    Truth     |        Outputs");

         for (int k = 0; k < layerSizes[A_INDEX]; k++)
         {
            System.out.print("-----|");
         }

         System.out.println("--------------|-----------------------");

         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            for (int k = 0; k < layerSizes[A_INDEX]; k++)
            {               
               if (testCases[testCase][k] >= 0) // Adds a space before positive values to ensure alignment
               {
                  System.out.print(" ");
               }

               System.out.print(String.format("%.2f", testCases[testCase][k]) + "|");
            } // for (int k = 0; k < layerSizes[A_INDEX]; k++)

            for (int i = 0; i < layerSizes[F_INDEX]; i++)
            {
               System.out.print(truthValues[testCase][i] + " |");
            }

            for (int i = 0; i < layerSizes[F_INDEX]; i++)
            {
               System.out.print(String.format("%,.4f", outputs[testCase][i]) + " |");
            }

            System.out.println();
         } // for (int testCase = 0; testCase < numTestCases; testCase++)
      } // if (printTruth)
      else
      {
         for (int k = 0; k < layerSizes[A_INDEX]; k++)
         {
            System.out.print("a" + k + "  |");
         }

         System.out.println("        Outputs");

         for (int k = 0; k < layerSizes[A_INDEX]; k++)
         {
            System.out.print("----|");
         }

         System.out.println("-----------------------");

         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            for (int k = 0; k < layerSizes[A_INDEX]; k++)
            {
               if (testCases[testCase][k] >= 0) // Adds a space before non-negative values so all values line up when printed
               {
                  System.out.print(" ");
               }

               System.out.print(String.format("%.2f", testCases[testCase][k]) + "|");
            } // for (int k = 0; k < layerSizes[A_INDEX]; k++)

            for (int i = 0; i < layerSizes[F_INDEX]; i++)
            {
               System.out.print(String.format("%,.4f", outputs[testCase][i]) + " |");
            }
            System.out.println();
         } // for (int testCase = 0; testCase < numTestCases; testCase++)
      } // else: if (printTruth)

      if (printWeights)
      {
         System.out.println("\nWeights:");

         for (int n = A_INDEX; n < layers; n++)
         {
            for (int j = 0; j < layerSizes[n]; j++) 
            {
               for (int k = 0; k < layerSizes[n + 1]; k++) 
               {
                  System.out.println("w" + n + "_" + j + "_" + k + ": " + String.format("%,.4f", weights[n][j][k]));
               }
            }
            System.out.println("\n=================================================");
         } // for (int n = A_INDEX; n < layers; n++)
         System.out.println();
      } // if (printWeights)
   } // void reportResults()

/**
 * Creates, configures, runs/trains, and reports the output of the neural network. Modifiable parameters allow the user to 
 * configure the network's operation and choose between running and training.
 */
   public static void main(String[] args)
   {
      Network net = new Network();
      if (args.length > 0)
      {
         net.configLoadPath = args[0];
      }
      else
      {
         net.configLoadPath = "control.json";
      }

      try 
      {
         net.loadConfigurationParameters();
         net.echoConfigurationParameters();
         net.allocateMemory();
         net.populateArrays();
         net.runOrTrain();
         if (net.saveWeights) 
         {
            net.saveWeights();
         }
         net.reportResults();
      } // try
      catch (Exception e) 
      {
         System.out.println(e);
         System.out.println("Because of this exception, the network wasn't able to " + (net.train ? "train" : "run") + ".");
      } // catch (Exception e)
   } // public static void main(String[] args) throws FileNotFoundException, IOException
} // public class Network