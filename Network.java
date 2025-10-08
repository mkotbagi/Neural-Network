import java.io.*;

/**
 * Author: Mihir Kotbagi
 * Date of creation: 9/5/25
 * Date of most recent modification: 10/9/25
 * Description: The Network class implements an A-B-C neural network written against the "Minimizing the Error Function" 
 * design document.
 */

public class Network 
{

   final static int MANUAL_TWO_TWO_ONE = 0; // Set weightPop to MANUAL_TWO_TWO_ONE to populate the weight array with 2-2-1 presets
   final static int RAND = 1; // Set weightPop to RAND to randomly populate the weight array
   final static int FILE_LOAD = 2; // Set weightPop to FILE_LOAD to populate the weight array from a file

   int aNodes; // Number of input activations
   int hNodes; // Number of hidden activations
   int fNodes; // Number of output nodes

   double[] a;   // 1D array representing all input activations
   double[] h;   // 1D array representing all hidden activations
   double[] F;  // 1D array representing all output activations

   double[][] weightKJ;
   double[][] weightJI;

   double lambda; // Learning factor

   int numTestCases; // Number of test cases

   double[][] testCases;   // 2D array storing test inputs
   double[][] truthValues; // 2D array storing truth values for the test cases

   double maxErrorThreshold; // Maximum acceptable average error across all test cases while training

   int weightPop;         // As described above, weightPop is set to 0, 1, or 2 depending on population method
   double randLow;        // Minimum value for randomization function
   double randHigh;       // Maximum value for randomization function
   String weightLoadPath; // Path to file weights should be loaded from
   String weightSavePath; // Path to file weights should be saved to

   boolean train;         // True if the network should be trained, false if the network should only be run
   boolean printTruth;    // True if the user wants to output the truth table, false otherwise
   boolean printWeights;  // True if the user wants to output the weight arrays, false otherwise
   boolean saveWeights;   // True if the user wants to save the weights, false otherwise

   int maxIterations; // Maximum number of iterations before network quits training

   int iterations;      // Number of iterations used for training
   double averageError; // Average error from the previous training iteration

   double[] Theta_i;
   double[] Theta_j;

   double[] Omega_j;
   double omega_i;

   double[] Psi_j; 
   double[] psi_i;

   double[][] partialE_wkj;
   double[][] partialE_wji;

   double[][] delta_wkj; // Weight change values for the weightKJ array (n = 1 layer)
   double[][] delta_wji; // Weight change values for the weightJ0 array (n = 2 layer)
   
   double[][] outputs; // Stores outputs for each test case to report to the user (distinct from output layer of activations)

/**
 * Allows the user to set parameters that configure the network and decide between training and running. Also allows the user
 * to specify if they want to print supplementary information (truth table and weights).
 */
   void setConfigurationParameters() 
   {
      aNodes = 2;
      hNodes = 1;
      fNodes = 3;

      train = false;

      numTestCases = 4;

      lambda = 0.3;
      maxErrorThreshold = 0.0002;
      maxIterations = 100000;

      weightPop = FILE_LOAD;
      randLow = 0.1;
      randHigh = 1.5;

      printTruth = false;
      printWeights = false;
      saveWeights = false;
      weightSavePath = "weights.bin";
      weightLoadPath = "weights.bin";
   } // void setConfigurationParameters()

/**
 * Prints out all the configuration parameters specified by the user.
 */
   void echoConfigurationParameters() 
   {
      System.out.println("=====================================================================");
      System.out.println("Network configuration: " + aNodes + "-" + hNodes + "-" + fNodes);
      System.out.println("This network has " + aNodes + " input activations, " + hNodes + " hidden activations, and "
            + fNodes + " output activation(s).");
      
      if (train) 
      {
         switch (weightPop) 
         {
            case RAND:
               System.out.println("The weights will be populated randomly, and the range for random weight population is ["
                     + randLow + ", " + randHigh + "].");
               break;
            case MANUAL_TWO_TWO_ONE:
               System.out.println("The weights will be populated manually using preset values for a 2-2-1 network.");
               break;
         } // switch (weightPop)
         System.out.println("Network is training against AND, OR, and XOR. \n\nTraining parameters:");
         System.out.println("The maximum number of iterations is " + maxIterations + ".");
         System.out.println("The maximum error threshold is " + maxErrorThreshold + ".");
         System.out.println("The learning factor is " + lambda + ".");
      } // if (train)
      else 
      {
         System.out.println("Network is running against AND, OR, and XOR.");
      }
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
         Theta_i = new double[fNodes];
         
         Omega_j = new double[hNodes];
         Psi_j = new double[hNodes];
         psi_i = new double[fNodes];

         partialE_wkj = new double[aNodes][hNodes];
         partialE_wji = new double[hNodes][fNodes];

         delta_wkj = new double[aNodes][hNodes];
         delta_wji = new double[hNodes][fNodes];
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
         populateTruth();
      }
      
      populateTests();
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
         for (int k = 0; k < aNodes; k++) {
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

      int checkA, checkH, checkF; // Used to verify that the weights match the network configuration
      checkA = dataIn.readInt();
      checkH = dataIn.readInt();
      checkF = dataIn.readInt();
      if (checkA != aNodes || checkH != hNodes || checkF != fNodes)
      {
         dataIn.close();
         throw new IOException("Weights from file don't match network configuration");
      }
      else
      {
         for(int j = 0; j < hNodes; j++)
         {
            for (int k = 0; k < aNodes; k++)
            {
               weightKJ[k][j] = dataIn.readDouble();
            }
         }
      
         for(int i = 0; i < fNodes; i++)
         {
            for (int j = 0; j < hNodes; j++) {
               weightJI[j][i] = dataIn.readDouble();
            }
         }
         
         dataIn.close();
      }
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

      for(int j = 0; j < hNodes; j++)
      {
         for (int k = 0; k < aNodes; k++)
         {
            dataOut.writeDouble(weightKJ[k][j]);
         }
      }
      
      for(int i = 0; i < fNodes; i++)
      {
         for (int j = 0; j < hNodes; j++) 
         {
            dataOut.writeDouble(weightJI[j][i]);
         }
      }
      
      dataOut.close();
   } // void saveWeights() throws FileNotFoundException, IOException
   
/**
 * Populates the truth table depending on which binary problem the user specifies
 */
   void populateTruth() 
   {
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
   } // void populateTruth()

/**
 * Populates the standard test cases for binary problems
 */
   void populateTests() 
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
 * Returns a random double precision floating point number between low and high.
 * Low must be less than or equal to high.
 */
   double randomize(double low, double high) 
   {
      return (high - low) * Math.random() + low;
   }

/**
 * Basic linear activation function f(x) = x
 */
   double oldActivationFunction(double x) 
   {
      return x;
   } // double oldActivationFunction(double x)

/**
 * Derivative of linear activation function f(x) = x; f'(x) = 1
 */
   double oldActivationFunctionDerivative(double x)
   {
      return 1.0;
   } // double oldActivationFunctionDerivative(double x)

/**
 * Sigmoid activation function
 */
   double activationFunction(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } // double activationFunction(double x)

/**
 * Derivative of the sigmoid activation function
 */
   double activationFunctionDerivative(double x)
   {
      double fX = activationFunction(x);
      return fX * (1.0 - fX);
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
      if (train)
      {
         iterations = 0;
         averageError = Double.MAX_VALUE; // The error must start above the threshold, otherwise training will not happen
         
         while (iterations < maxIterations && averageError > maxErrorThreshold) 
         {
            averageError = 0.0;
            
            for (int testCase = 0; testCase < numTestCases; testCase++) 
            {
               runForTrain(testCase);
               calcGradientDescent(testCase);
               gradientDescent();
            } // for (int testCase = 0; testCase < numTestCases; testCase++)
            
            iterations++;
/**
 * averageError is summed over all testcases in the above loop through the calcGradientDescent(testCase) method, so it must be 
 * divided by numTestCases to obtain the actual average
 */ 
            averageError /= (double) numTestCases;
         } // while (iterations < maxIterations && averageError > maxErrorThreshold)

         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            run(testCase);
            for (int i = 0; i < fNodes; i++)
            {
               outputs[testCase][i] = F[i];
            }
         }

         if(saveWeights)
         {
            saveWeights();
         }

      } // if (train)
      else
      {
         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            run(testCase);
            for (int i = 0; i < fNodes; i++)
            {
               outputs[testCase][i] = F[i];
            }
         }
      }
   } // void runOrTrain()
   
/**
 * Runs the network for a particular test case; dot products aren't stored because they aren't needed for running without 
 * training.
 */
   void run(int testCase)
   {
      populateInputActivations(testCase);

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
         for(int j = 0; j < hNodes; j++)
         {
            Theta += h[j] * weightJI[j][i];
         }
         F[i] = activationFunction(Theta);
      } // for(int i = 0; i < fNodes; i++)
   } // void run(int testCase)

/**
 * Runs the network for a particular test case and stores all dot products.
 */
   void runForTrain(int testCase)
   {
      populateInputActivations(testCase);

      for (int j = 0; j < hNodes; j++)
      {
         Theta_j[j] = 0.0;

         for (int k = 0; k < aNodes; k++)
         {
            Theta_j[j] += a[k] * weightKJ[k][j];
         }

         h[j] = activationFunction(Theta_j[j]);
      } // for (int j = 0; j < hNodes; j++)
      
      for (int i = 0; i < fNodes; i++)
      {
         Theta_i[i] = 0.0;
         for (int j = 0; j < hNodes; j++)
         {
            Theta_i[i] += h[j] * weightJI[j][i];
         }
         F[i] = activationFunction(Theta_i[i]);
      } // for(int i = 0; i < fNodes; i++)
   } // void runForTrain(int testCase)

/**
 * Populates the delta weight arrays, which are used to update the weights while training.
 */
   void calcGradientDescent(int testCase)
   {
      for (int i = 0; i < fNodes; i++)
      {
         omega_i = truthValues[testCase][i] - F[i];
         psi_i[i] = omega_i * activationFunctionDerivative(Theta_i[i]);
         averageError += omega_i * omega_i / 2.0;
      }

      for (int j = 0; j < hNodes; j++)
      {
         Omega_j[j] = 0.0;

         for (int i = 0; i < fNodes; i++)
         {
            Omega_j[j] += psi_i[i] * weightJI[j][i];
            partialE_wji[j][i] = -h[j] * psi_i[i];
            delta_wji[j][i] = -lambda * partialE_wji[j][i];
         }

         Psi_j[j] = Omega_j[j] * activationFunctionDerivative(Theta_j[j]);

         for (int k = 0; k < aNodes; k++)
         {
            partialE_wkj[k][j] = -a[k] * Psi_j[j];
            delta_wkj[k][j] = -lambda * partialE_wkj[k][j];
         }
      } // for (int j = 0; j < hNodes; j++)
   } // void calcGradientDescent(int testCase)
   
/**
 * Updates all weights using the calculated weight change arrays.
 */
   void gradientDescent()
   {
      for (int j = 0; j < hNodes; j++) 
      {
         for (int k = 0; k < aNodes; k++) 
         {
            weightKJ[k][j] += delta_wkj[k][j];
         }

         for(int i = 0; i < fNodes; i++)
         {
            weightJI[j][i] += delta_wji[j][i];
         }
      } // for (int j = 0; j < hNodes; j++)
   } // void gradientDescent()
   
/**
 * Reports the results for running or training, and depending on configuration parameters, outputs test cases and weights as well.
 */
   void reportResults()
   {
      System.out.println("Network attempted to " + (train ? "train" : "run") + " on " + numTestCases + " test cases.");
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

         if(saveWeights)
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

         System.out.println("-------------------------------------------------------");

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
            for(int i = 0; i < fNodes; i++)
            {
               System.out.print(String.format("%,.10f", outputs[testCase][i]) + " |");
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
            for(int i = 0; i < fNodes; i++)
            {
               System.out.print(String.format("%,.10f", outputs[testCase][i]) + " |");
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
      net.setConfigurationParameters();
      net.echoConfigurationParameters();
      net.allocateMemory();
      net.populateArrays();
      net.runOrTrain();
      net.reportResults();
   } // public static void main(String[] args) throws FileNotFoundException, IOException
} // public class Network