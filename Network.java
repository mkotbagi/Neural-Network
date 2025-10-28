import java.io.*;
import java.util.*;
import com.google.gson.*;

/**
 * Author: Mihir Kotbagi
 * Date of creation: 9/5/25
 * Date of most recent modification: 10/28/25
 * Description: The Network class implements an A-B-C neural network that uses backpropagation for training. It is written against
 * the "Minimizing and Optimizing the Error Function" design document. Configuration parameters are read from a user-selected file
 * using Google's GSON library for JSON parsing.
 */
public class Network 
{

   final static int MANUAL_TWO_TWO_ONE = 0; // Set weightPop to MANUAL_TWO_TWO_ONE to populate the weight array with 2-2-1 presets
   final static int RAND = 1;               // Set weightPop to RAND to randomly populate the weight array
   final static int FILE_LOAD = 2;          // Set weightPop to FILE_LOAD to populate the weight array from a file

   int aNodes;                // Number of input activations
   int hNodes;                // Number of hidden activations
   int fNodes;                // Number of output nodes

   double[] a;                // 1D array representing all input activations
   double[] h;                // 1D array representing all hidden activations
   double[] F;                // 1D array representing all output activations

   double[][] weightKJ;
   double[][] weightJI;

   double lambda;             // Learning factor

   int numTestCases;          // Number of test cases

   double[][] testCases;      // 2D array storing test inputs
   double[][] truthValues;    // 2D array storing truth values for the test cases

   String test;               // "AND", "OR", "XOR", or "AND_OR_XOR" depending on what the user wants to run/train against

   double maxErrorThreshold;  // Maximum acceptable average error across all test cases while training

   int weightPop;             // As described above, weightPop is set to 0, 1, or 2 depending on population method
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

   int iterations;            // Number of iterations used for training
   double averageError;       // Average error from the previous training iteration

   double[] Theta_j;
   double[] psi_i;
   
   double[][] outputs;        // Stores outputs for each test case to report (distinct from output layer of activations)

   long runOrTrainStart;      // Used to time running/training
   long runOrTrainFinish;     // Used to time running/training

/**
 * Allows the user to set parameters that configure the network and decide between training and running. Also allows the user
 * to specify if they want to print supplementary information (truth table and weights).
 */
   void setConfigurationParameters() 
   {
      aNodes = 2;
      hNodes = 5;
      fNodes = 3;

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
      configLoadPath = "control.json";
      FileReader reader = new FileReader(configLoadPath);

      JsonObject parameters = JsonParser.parseReader(reader).getAsJsonObject();

      aNodes = parameters.get("aNodes").getAsInt();
      hNodes = parameters.get("hNodes").getAsInt();
      fNodes = parameters.get("fNodes").getAsInt();

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
         case "MANUAL_TWO_TWO_ONE":
            weightPop = MANUAL_TWO_TWO_ONE;
            break;
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
   } // void loadConfigurationParameters() throws FileNotFoundException, IOException
   
/**
 * Prints out all the configuration parameters specified by the user.
 */
   void echoConfigurationParameters() 
   {
      System.out.println("=====================================================================");
      System.out.println("Network configuration: " + aNodes + "-" + hNodes + "-" + fNodes);
      System.out.println("This network has " + aNodes + " input activations, " + hNodes + " hidden activations, and "
            + fNodes + " output activation(s).");
      
      switch (weightPop) 
      {
         case RAND:
            System.out.println("The weights will be populated randomly, and the range for random weight population is ["
                  + randLow + ", " + randHigh + "].");
            break;
         case MANUAL_TWO_TWO_ONE:
            System.out.println("The weights will be populated manually using preset values for a 2-2-1 network.");
            break;
         case FILE_LOAD:
            System.out.println("The weights were loaded from " + weightSavePath + ".");
            break;
      } // switch (weightPop)

      if (train) 
      {
         System.out.println("Network is training against " + test + ". \n\nTraining parameters:");
         System.out.println("The maximum number of iterations is " + maxIterations + ".");
         System.out.println("Weights will be saved every " + weightSaveFreq + " iterations.");
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
      a = new double[aNodes];
      h = new double[hNodes];
      F = new double[fNodes];

      weightKJ = new double[aNodes][hNodes];
      weightJI = new double[hNodes][fNodes];

      testCases = new double[numTestCases][aNodes];

      outputs = new double[numTestCases][fNodes];

      if (train || printTruth) // The truth table is only needed if the user is training or wants to print it
      {
         truthValues = new double[numTestCases][fNodes];  
      }

      if (train) 
      {
         Theta_j = new double[hNodes];
         psi_i = new double[fNodes];
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
      else if (weightPop == MANUAL_TWO_TWO_ONE) 
      {
         manualPopulateWeights();
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
      } // if (loadTests)
   } // void populateArrays() throws FileNotFoundException, IOException
   
/**
 * Populates the weight arrays using preset values for a 2-2-1 network
 */
   void manualPopulateWeights() 
   {
      weightKJ[0][0] = 0.45;
      weightKJ[1][0] = 0.45;
      weightKJ[0][1] = 0.45;
      weightKJ[1][1] = 0.45;

      weightJI[0][0] = 0.66;
      weightJI[1][0] = 0.66;
   } // void manualPopulateWeights()
   
/**
 * Populates the weight arrays randomly; each weight is set to a random double precision value between randLow and randHigh
 */
   void randomPopulateWeights() 
   {
      for (int j = 0; j < hNodes; j++) 
      {
         for (int k = 0; k < aNodes; k++) 
         {
            weightKJ[k][j] = randomize(randLow, randHigh);
         }
      }
      
      for (int i = 0; i < fNodes; i++) 
      {
         for (int j = 0; j < hNodes; j++) 
         {
            weightJI[j][i] = randomize(randLow, randHigh);
         }
      }
   } // void randomPopulateWeights()

/**
 * Loads the weights from a file
 */
   void loadWeights() throws FileNotFoundException, IOException
   {
      File weightFile = new File(weightLoadPath);
      DataInputStream dataIn = new DataInputStream(new FileInputStream(weightFile));

      int aNodesW, hNodesW, fNodesW; // Used to verify that the config saved in the weights file matches the user config
      aNodesW = dataIn.readInt();
      hNodesW = dataIn.readInt();
      fNodesW = dataIn.readInt();

      if (aNodesW != aNodes || hNodesW != hNodes || fNodesW != fNodes)
      {
         dataIn.close();
         System.out.println("Weights from file don't match network configuration, so they couldn't be loaded.");
         System.out.println("Populating weights randomly instead.");
         randomPopulateWeights();
      } // if (aNodesW != aNodes || hNodesW != hNodes || fNodesW != fNodes)
      else
      {
         for (int j = 0; j < hNodes; j++)
         {
            for (int k = 0; k < aNodes; k++)
            {
               weightKJ[k][j] = dataIn.readDouble();
            }
         }
      
         for (int i = 0; i < fNodes; i++)
         {
            for (int j = 0; j < hNodes; j++) 
            {
               weightJI[j][i] = dataIn.readDouble();
            }
         }
         
         dataIn.close();
      } // else
   } // void loadWeights() throws FileNotFoundException, IOException

/**
 * Saves the weights to a file in binary
 */
   void saveWeights() throws FileNotFoundException, IOException
   {
      File weightFile = new File(weightSavePath);
      DataOutputStream dataOut = new DataOutputStream(new FileOutputStream(weightFile));

      dataOut.writeInt(aNodes); // First saves network configuration, which is checked when loading weights from a file
      dataOut.writeInt(hNodes);
      dataOut.writeInt(fNodes);

      for (int j = 0; j < hNodes; j++)
      {
         for (int k = 0; k < aNodes; k++)
         {
            dataOut.writeDouble(weightKJ[k][j]);
         }
      } // for (int j = 0; j < hNodes; j++)
      
      for (int i = 0; i < fNodes; i++)
      {
         for (int j = 0; j < hNodes; j++) 
         {
            dataOut.writeDouble(weightJI[j][i]);
         }
      } // for (int i = 0; i < fNodes; i++)
      
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
         for (int i = 0; i < fNodes; i++) 
         {
            truthValues[testCase][i] = scanner.nextInt();
         }
      }
      
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
   } // void populateTests()

/**
 * Loads test cases from a user-specified file
 */   
   void loadTests() throws FileNotFoundException, IOException
   {
      File testFile = new File(testCaseLoadPath);
      Scanner scanner = new Scanner(testFile);

      for (int testCase = 0; testCase < numTestCases; testCase++)
      {
         for (int k = 0; k < aNodes; k++) 
         {
            testCases[testCase][k] = scanner.nextInt();
         }
      }
      
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
   } // double oldActivationFunction(double x)

/**
 * Derivative of linear activation function f(x) = x; f'(x) = 1
 */
   double linearDerivative(double x)
   {
      return 1.0;
   } // double oldActivationFunctionDerivative(double x)

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
 * epsilon(x) is a helper function for the tanh activation function
 */
   double epsilon(double x)
   {
      return (x < 0) ? 1.0 : -1.0;
   } // double epsilon(double x)

/**
 * Returns the value of the hyperbolic tangent function for a given value of x
 */
   double tanh(double x)
   {
      double epsX = epsilon(x);
      double e_eps_2X = Math.exp(epsX * 2.0 * x);
      return epsX * (e_eps_2X - 1.0) / (e_eps_2X + 1.0);
   } // double tanh(double x)

/**
 * Returns the value of the derivative of the hyperbolic tangent function for a given value of x
 */
   double tanhDerivative(double x)
   {
      double fX = tanh(x);
      return 1 - fX * fX;
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
      for (int k = 0; k < aNodes; k++)
      {
         a[k] = testCases[testCase][k];
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

            for (int i = 0; i < fNodes; i++)
            {
               outputs[testCase][i] = F[i];
            }
         } // for (int testCase = 0; testCase < numTestCases; testCase++)
      } // if (train)
      else
      {
         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            populateInputActivations(testCase);
            run(testCase);
            
            for (int i = 0; i < fNodes; i++)
            {
               outputs[testCase][i] = F[i];
            }
         } // for (int testCase = 0; testCase < numTestCases; testCase++)
      } // else

      runOrTrainFinish = System.nanoTime();
   } // void runOrTrain()
   
/**
 * Runs the network for a particular test case; dot products aren't stored because they aren't needed for running without 
 * training.
 */
   void run(int testCase)
   {
      double Theta;

      for (int j = 0; j < hNodes; j++)
      {
         Theta = 0.0;

         for (int k = 0; k < aNodes; k++)
         {
            Theta += a[k] * weightKJ[k][j];
         }

         h[j] = activationFunction(Theta);
      } // for (int j = 0; j < hNodes; j++)
      
      for (int i = 0; i < fNodes; i++)
      {
         Theta = 0.0;

         for (int j = 0; j < hNodes; j++)
         {
            Theta += h[j] * weightJI[j][i];
         }

         F[i] = activationFunction(Theta);
      } // for (int i = 0; i < fNodes; i++)
   } // void run(int testCase)

/**
 * Runs the network for a particular test case and stores all dot products.
 */
   double runForTrain(int testCase)
   {
      for (int j = 0; j < hNodes; j++)
      {
         Theta_j[j] = 0.0;

         for (int k = 0; k < aNodes; k++)
         {
            Theta_j[j] += a[k] * weightKJ[k][j];
         }

         h[j] = activationFunction(Theta_j[j]);
      } // for (int j = 0; j < hNodes; j++)
      
      Double error = 0.0;

      for (int i = 0; i < fNodes; i++)
      {
         Double Theta_i = 0.0;
         
         for (int j = 0; j < hNodes; j++)
         {
            Theta_i += h[j] * weightJI[j][i];
         }
         
         F[i] = activationFunction(Theta_i);

         double omega_i = truthValues[testCase][i] - F[i];
         psi_i[i] = omega_i * activationFunctionDerivative(Theta_i);
         error += omega_i * omega_i;
      } // for (int i = 0; i < fNodes; i++)

      return error;
   } // void runForTrain(int testCase)

/**
 * Updates the weights after calculating how much they should change
 */
   void gradientDescent(int testCase)
   {
      for (int j = 0; j < hNodes; j++)
      {
         double Omega_j = 0.0;

         for (int i = 0; i < fNodes; i++)
         {
            Omega_j += psi_i[i] * weightJI[j][i];
            weightJI[j][i] += lambda * h[j] * psi_i[i];
         }

         double Psi_j = Omega_j * activationFunctionDerivative(Theta_j[j]);

         for (int k = 0; k < aNodes; k++)
         {
            weightKJ[k][j] += lambda * a[k] * Psi_j;
         }
      } // for (int j = 0; j < hNodes; j++)
   } // void calcGradientDescent(int testCase)
   
/**
 * Reports the results for running or training, and depending on configuration parameters, outputs test cases and weights as well.
 */
   void reportResults()
   {
      System.out.println("Network attempted to " + (train ? "train" : "run") + " on " + numTestCases + " test cases.");
      System.out.println((train ? "Training" : "Running") + " took " + (runOrTrainFinish - runOrTrainStart) / 1e9 + " seconds.");

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

         System.out.println("\nAverage Error: " + averageError);
         System.out.println("Iterations: " + iterations);

         if (saveWeights)
         {
            System.out.println("Weights saved to " + weightSavePath);
         }
      } // if (train)

      System.out.println("\nResults:");
      if (printTruth)
      {
         for (int k = 0; k < aNodes; k++)
         {
            System.out.print("a" + k + "  |");
         }
         System.out.println("    Truth     |                Outputs");

         for (int k = 0; k < aNodes; k++)
         {
            System.out.print("----|");
         }

         System.out.println("--------------|----------------------------------------");

         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            for (int k = 0; k < aNodes; k++)
            {
               System.out.print(testCases[testCase][k] + " |");
            }

            for (int i = 0; i < fNodes; i++)
            {
               System.out.print(truthValues[testCase][i] + " |");
            }

            for (int i = 0; i < fNodes; i++)
            {
               System.out.print(String.format("%,.17f", outputs[testCase][i]) + " |");
            }

            System.out.println();
         } // for (int testCase = 0; testCase < numTestCases; testCase++)
      } // if (printTruth)
      else
      {
         for (int k = 0; k < aNodes; k++)
         {
            System.out.print("a" + k + "  |");
         }

         System.out.println("               Outputs");

         for (int k = 0; k < aNodes; k++)
         {
            System.out.print("----|");
         }

         System.out.println("----------------------------------------");

         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            for (int k = 0; k < aNodes; k++)
            {
               System.out.print(testCases[testCase][k] + " |");
            }

            for (int i = 0; i < fNodes; i++)
            {
               System.out.print(String.format("%,.17f", outputs[testCase][i]) + " |");
            }
            System.out.println();
         } // for (int testCase = 0; testCase < numTestCases; testCase++)
      } // else

      if (printWeights)
      {
         System.out.println("\nWeights:");
         for (int j = 0; j < hNodes; j++)
         {
            for (int k = 0; k < aNodes; k++) 
            {
               System.out.println("w1_" + k + "_" + j + ": " + weightKJ[k][j]);
            }

            for (int i = 0; i < fNodes; i++)
            {
               System.out.println("w2_" + j + "_" + i + ": " + Double.toString(weightJI[j][i]));
            }
            System.out.println("\n=================================================");
         } // for (int j = 0; j < hNodes; j++)

         System.out.println();
      } // if (printWeights)
   } // void reportResults()

/**
 * Creates, configures, runs/trains, and reports the output of the neural network. Modifiable parameters allow the user to 
 * configure the network's operation and choose between running and training.
 */
   public static void main(String[] args) throws FileNotFoundException, IOException
   {
      Network net = new Network();
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
         System.out.println(e.getMessage());
         System.out.println("Because of this exception, the network wasn't able to " + (net.train ? "train" : "run") + ".");
      } // catch (Exception e)
   } // public static void main(String[] args) throws FileNotFoundException, IOException
} // public class Network