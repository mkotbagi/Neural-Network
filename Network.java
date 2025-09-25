/**
 * Author: Mihir Kotbagi
 * Date of creation: 9/5/25
 * Date of most recent modification: 9/23/25
 * Description: The Network class represents an A-B-1 neural network and contains methods for configuring, running, and training * the network. This class is written against the "Minimization of the Single Output Error Function" design document.
 */

public class Network 
{

   final static int MANUAL_TWO_TWO_ONE = 0; // To manually populate the weight array with preset values for a 2-2-1 network, set 
                                            // weightPop to MANUAL_TWO_TWO_ONE
   
   final static int RAND = 1; // To randomly populate the weight array, set weightPop to RAND

   int aNodes; // Number of input activations
   int hNodes; // Number of hidden activations
   int fNodes; // Number of output nodes (1 for A-B-1 network)

   double[] a; // 1D array representing all input activations
   double[] h; // 1D array representing all hidden activations
   double F0;  // Output activation (only 1 element for A-B-1)

   double[][] weightKJ;
   double[] weightJ0;

   double lambda; // Learning factor

   int numTestCases; // Number of test cases

   double[][] testCases; // 2D array storing test inputs
   double[] truthValues; // 1D array storing truth values for the test cases

   double maxErrorThreshold; // Maximum acceptable average error across all test cases while training

   int weightPop;        // MANUAL_TWO_TWO_ONE (0) to populate randomly, RAND (1) to populate using the preset values
   double randLow;       // Minimum value for randomization function
   double randHigh;      // Maxmum value for randomization function
   boolean train;        // True if the network should be trained, false if the network should only be run
   boolean printTruth;   // True if the user wants to output the truth table, false otherwise
   boolean printWeights; // True if the user wants to output the weight arrays, false otherwise

   String test; // Test ("AND", "OR", "XOR") to use

   int maxIterations; // Maximum number of iterations before network quits training

   int iterations;      // Number of iterations used for training
   double averageError; // Average error from the previous training iteration

   double[] Theta_j;
   double Theta_0;

   double[] Omega_j;
   double omega_0;

   double[] Psi_j; 
   double psi_0;

   double[][] partialE_wkj;
   double[] partialE_wj0;

   double[][] delta_wkj; // Weight change values for the weightKJ array (n = 1 layer)
   double[] delta_wj0;   // Weight change values for the weightJ0 array (n = 2 layer)
   
   double[] outputs; // Output values for each test case to report to the user

   /**
    * Allows the user to set parameters that configure the network and decide between training and running. Also allows the user
    * to specify if they want to print supplementary information (truth table and weights).
    */
   void setConfigurationParameters() 
   {
      aNodes = 2;
      hNodes = 1;
      fNodes = 1;

      test = "XOR";

      train = false;

      numTestCases = 4;

      lambda = 0.3;
      maxErrorThreshold = 0.0002;
      maxIterations = 100000;

      weightPop = RAND;
      randLow = -1.5;
      randHigh = 1.5;

      printTruth = true;
      printWeights = false;
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
         System.out.println("Network is training against " + test + ". \n\nTraining parameters:");
         System.out.println("The maximum number of iterations is " + maxIterations + ".");
         System.out.println("The maximum error threshold is " + maxErrorThreshold + ".");
         System.out.println("The learning factor is " + lambda + ".");
      } // if (train)
      else 
      {
         System.out.println("Network is running against " + test + ".");
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

      weightKJ = new double[aNodes][hNodes];
      weightJ0 = new double[hNodes];

      testCases = new double[numTestCases][aNodes];

      outputs = new double[numTestCases];

      if (train || printTruth) // The truth table is only needed if the user is training or wants to print it
      {
         truthValues = new double[numTestCases];  
      }

      if (train) 
      {
         Theta_j = new double[hNodes];
         
         Omega_j = new double[hNodes];
         Psi_j = new double[hNodes];

         partialE_wkj = new double[aNodes][hNodes];
         partialE_wj0 = new double[hNodes];

         delta_wkj = new double[aNodes][hNodes];
         delta_wj0 = new double[hNodes];
      } // if (train)
   } // void allocateMemory()

   /**
    * Populates the truth table and weight arrays according to the user specification. The truth table isn't populated if the user
    * isn't training and doesn't want to view it; the tests are always populated because they are used for training and running.
    */
   void populateArrays() 
   {
      if (weightPop == RAND) 
      {
         randomPopulateWeights();
      } 
      else if (weightPop == MANUAL_TWO_TWO_ONE) 
      {
         manualPopulateWeights();
      }
      if (train || printTruth)
         populateTruth();
      populateTests();
   } // void populateArrays()
   
   /**
    * Populates the weight arrays using preset values for a 2-2-1 network
    */
   void manualPopulateWeights() 
   {
      weightKJ[0][0] = 0.45;
      weightKJ[1][0] = 0.45;
      weightKJ[0][1] = 0.45;
      weightKJ[1][1] = 0.45;

      weightJ0[0] = 0.66;
      weightJ0[1] = 0.66;
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
         weightJ0[j] = randomize(randLow, randHigh);
      }
   } // void randomPopulateWeights()
   
   /**
    * Populates the truth table depending on which binary problem the user specifies
    */
   void populateTruth() 
   {
      switch (test) 
      {
         case "AND":
            truthValues[0] = 0.0;
            truthValues[1] = 0.0;
            truthValues[2] = 0.0;
            truthValues[3] = 1.0;
            break;
      
         case "OR":
            truthValues[0] = 0.0;
            truthValues[1] = 1.0;
            truthValues[2] = 1.0;
            truthValues[3] = 1.0;
            break;

         case "XOR":
            truthValues[0] = 0.0;
            truthValues[1] = 1.0;
            truthValues[2] = 1.0;
            truthValues[3] = 0.0;
            break;
      } // switch (test)
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
      return 1;
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
   void runOrTrain()
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
               runForTrain(testCase);
            } // for (int testCase = 0; testCase < numTestCases; testCase++)
            
            iterations++;
            averageError /= (double) numTestCases; // averageError is summed over all testcases in the above loop through the 
                                                   //calcGradientDescent(testCase) method, so it must be divided by
                                                   // numTestCases to get the actual average

         } // while (iterations < maxIterations && averageError > maxErrorThreshold)
      } // if (train)
      else
      {
         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            run(testCase);
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

      double Theta = 0.0;

      for (int j = 0; j < hNodes; j++)
      {
         Theta = 0.0;

         for (int k = 0; k < aNodes; k++)
         {
            Theta += a[k] * weightKJ[k][j];
         }

         h[j] = activationFunction(Theta);
      } // for (int j = 0; j < hNodes; j++)
      
      Theta = 0.0;

      for (int j = 0; j < hNodes; j++)
      {
         Theta += h[j] * weightJ0[j];
      }

      F0 = activationFunction(Theta);
      outputs[testCase] = F0;
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
      
      Theta_0 = 0.0;

      for (int j = 0; j < hNodes; j++)
      {
         Theta_0 += h[j] * weightJ0[j];
      }

      F0 = activationFunction(Theta_0);
      outputs[testCase] = F0;
   } // void runForTrain(int testCase)

   /**
    * Populates the delta weight arrays, which are used to update the weights while training.
    */
   void calcGradientDescent(int testCase)
   {
      omega_0 = truthValues[testCase] - F0;
      psi_0 = omega_0 * activationFunctionDerivative(Theta_0);

      averageError += omega_0 * omega_0 / 2.0; // averageError is divided by numTestCases in runOrTrain to obtain the average 
                                               // from the total over all testcases
      
      for (int j = 0; j < hNodes; j++)
      {
         Omega_j[j] = psi_0 * weightJ0[j];
         Psi_j[j] = Omega_j[j] * activationFunctionDerivative(Theta_j[j]);

         partialE_wj0[j] = -h[j] * psi_0;
         delta_wj0[j] = -lambda * partialE_wj0[j];

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
         weightJ0[j] += delta_wj0[j];

         for (int k = 0; k < aNodes; k++) 
         {
            weightKJ[k][j] += delta_wkj[k][j];
         }
      } // for (int j = 0; j < hNodes; j++)
   } // void gradientDescent()
   
   /**
    * Reports the results for running or training, and depending on configuration parameters, outputs the test cases and weights * as well.
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
      } // if (train)

      System.out.println("\nResults:");
      if (printTruth)
      {
         System.out.println("a0  | a1  | Truth | Output");
         System.out.println("----|-----|-------|-------");
         for (int testCase = 0; testCase < numTestCases; testCase++) 
         {
            System.out.println(testCases[testCase][0] + " | " + testCases[testCase][1] + " |  " + truthValues[testCase]
                  + "  | " + outputs[testCase]);
         }
      } // if (printTruth)
      else
      {
         System.out.println("a0  | a1  | Output");
         System.out.println("----|-----|-------");
         for (int testCase = 0; testCase < numTestCases; testCase++) 
         {
            System.out.println(testCases[testCase][0] + " | " + testCases[testCase][1] + " | " + outputs[testCase]);
         }
      }
      if (printWeights)
      {
         System.out.println("\nWeights:");
         for (int j = 0; j < hNodes; j++)
         {
            for (int k = 0; k < aNodes; k++) 
            {
               System.out.print("\nw1_" + k + "_" + j + ": " + weightKJ[k][j]);
            }
            System.out.print(" | w2_" + j + "_0: " + weightJ0[j]);
            System.out.println("\n=================================================");
         }
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
      net.setConfigurationParameters();
      net.echoConfigurationParameters();
      net.allocateMemory();
      net.populateArrays();
      net.runOrTrain();
      net.reportResults();
   } // public static void main(String[] args)
} // public class Network