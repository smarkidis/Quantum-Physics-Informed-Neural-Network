import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import random

import matplotlib.pyplot as plt

import scipy.optimize as sopt

tf.config.run_functions_eagerly(True)

# Four Layer QPINN - BATCHED

#collocation points
n_collocation_points = 128
Lx = 1.0
#Lx = 2*np.pi
dx = Lx / (n_collocation_points  - 1)
x_collocation_point = np.linspace(0, Lx , n_collocation_points)

# number of iterations
n_epochs = 500 # 500
rate = 0.01

# Set the optimizer parameters
#opt = tf.keras.optimizers.SGD(learning_rate=rate, name="SGD")  # 0.0025  / 0.01 for x^2 in  b  # sin2x --> learning_rate=0.0001
#opt = tf.keras.optimizers.Adam(learning_rate=rate, name="Adam") 

sdev = 0.05 #0.05

# Backend Quantum Computer Simulator
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 125, "batch_size": n_collocation_points})
prog = sf.Program(1)

seed = 11

# set the random seed for the TF and backend
tf.random.set_seed(seed)
np.random.seed(seed)

#*** Define TF parameters
x = prog.params("x")


# First Layer - Parameters
alpha1 = prog.params("alpha1")
phi1 = prog.params("phi1")

r1 = prog.params("r1")

sq_r1 = prog.params("sq_r1")
sq_phi1 = prog.params("sq_phi1")

r2 = prog.params("r2")

kappa1 = prog.params("kappa1")

# Second Layer - Parameters _2

alpha1_2 = prog.params("alpha1_2")
phi1_2 = prog.params("phi1_2")

r1_2 = prog.params("r1_2")

sq_r1_2 = prog.params("sq_r1_2")
sq_phi1_2 = prog.params("sq_phi1_2")

r2_2 = prog.params("r2_2")

kappa1_2 = prog.params("kappa1_2")

# Third Layer - Parameters _3

alpha1_3 = prog.params("alpha1_3")
phi1_3 = prog.params("phi1_3")

r1_3 = prog.params("r1_3")

sq_r1_3 = prog.params("sq_r1_3")
sq_phi1_3 = prog.params("sq_phi1_3")

r2_3 = prog.params("r2_3")

kappa1_3 = prog.params("kappa1_3")

# Fourth Layer - Parameters _4

alpha1_4 = prog.params("alpha1_4")
phi1_4 = prog.params("phi1_4")

r1_4 = prog.params("r1_4")

sq_r1_4 = prog.params("sq_r1_4")
sq_phi1_4 = prog.params("sq_phi1_4")

r2_4 = prog.params("r2_4")

kappa1_4 = prog.params("kappa1_4")



###################
##   QPINN Circuit
###################
with prog.context as q:
    ###################
    # Encode the data #
    ###################
    Dgate(x) | q

    

    ###############
    # First Layer #
    ###############
    # Displacement
    Dgate(alpha1, phi1) | q

    # First Rotation
    Rgate(r1) | q

    # Squeezeer
    Sgate(sq_r1, sq_phi1) | q

    # Second Rotation
    Rgate(r2) | q

    # Kerr gate
    Kgate(kappa1) | q

    
    
    #######################
    # Second Layer        #
    #######################
    # Displacement
    Dgate(alpha1_2, phi1_2) | q

    # First Rotation
    Rgate(r1_2) | q

    # Squeezeer
    Sgate(sq_r1_2, sq_phi1_2) | q

    # Second Rotation
    Rgate(r2_2) | q

    # Kerr gate
    Kgate(kappa1_2) | q

    
    
    #######################
    # Third Layer        #
    #######################
    # Displacement
    Dgate(alpha1_3, phi1_3) | q

    # First Rotation
    Rgate(r1_3) | q

    # Squeezeer
    Sgate(sq_r1_3, sq_phi1_3) | q

    # Second Rotation
    Rgate(r2_3) | q

    # Kerr gate
    Kgate(kappa1_3) | q


    #######################
    # Fourth Layer        #
    #######################
    # Displacement
    Dgate(alpha1_4, phi1_4) | q

    # First Rotation
    Rgate(r1_4) | q

    # Squeezeer
    Sgate(sq_r1_4, sq_phi1_4) | q

    # Second Rotation
    Rgate(r2_4) | q

    # Kerr gate
    Kgate(kappa1_4) | q

    
# Assign our TensorFlow variables, so that we can
# refer to them later when differentiating/training.

sdev = 0.05 #0.05


# Circuit parameters

# First Layer
alpha1_in= tf.Variable(tf.random.normal(shape=[], stddev=sdev))
phi1_in= tf.Variable(tf.random.normal(shape=[], stddev=sdev))

r1_in= tf.Variable(tf.random.normal(shape=[], stddev=sdev))

sq_r1_in = tf.Variable(tf.random.normal(shape=[], stddev=sdev))
sq_phi1_in = tf.Variable(tf.random.normal(shape=[], stddev=sdev))

r2_in= tf.Variable(tf.random.normal(shape=[], stddev=sdev))

kappa1_in= tf.Variable(tf.random.normal(shape=[], stddev=sdev))

# Second Layer
alpha1_in_2= tf.Variable(tf.random.normal(shape=[], stddev=sdev))
phi1_in_2= tf.Variable(tf.random.normal(shape=[], stddev=sdev))

r1_in_2= tf.Variable(tf.random.normal(shape=[], stddev=sdev))

sq_r1_in_2 = tf.Variable(tf.random.normal(shape=[], stddev=sdev))
sq_phi1_in_2 = tf.Variable(tf.random.normal(shape=[], stddev=sdev))

r2_in_2= tf.Variable(tf.random.normal(shape=[], stddev=sdev))

kappa1_in_2= tf.Variable(tf.random.normal(shape=[], stddev=sdev))


# Third Layer
alpha1_in_3= tf.Variable(tf.random.normal(shape=[], stddev=sdev))
phi1_in_3= tf.Variable(tf.random.normal(shape=[], stddev=sdev))

r1_in_3= tf.Variable(tf.random.normal(shape=[], stddev=sdev))

sq_r1_in_3 = tf.Variable(tf.random.normal(shape=[], stddev=sdev))
sq_phi1_in_3 = tf.Variable(tf.random.normal(shape=[], stddev=sdev))

r2_in_3 = tf.Variable(tf.random.normal(shape=[], stddev=sdev))

kappa1_in_3 = tf.Variable(tf.random.normal(shape=[], stddev=sdev))


# Fourth Layer
alpha1_in_4= tf.Variable(tf.random.normal(shape=[], stddev=sdev))
phi1_in_4= tf.Variable(tf.random.normal(shape=[], stddev=sdev))

r1_in_4= tf.Variable(tf.random.normal(shape=[], stddev=sdev))

sq_r1_in_4 = tf.Variable(tf.random.normal(shape=[], stddev=sdev))
sq_phi1_in_4 = tf.Variable(tf.random.normal(shape=[], stddev=sdev))

r2_in_4 = tf.Variable(tf.random.normal(shape=[], stddev=sdev))

kappa1_in_4 = tf.Variable(tf.random.normal(shape=[], stddev=sdev))



# regularization                                                                                                                                                    
regularization = 0.0 #0.5
reg_variance = 0.0 #2.0 #10.0

reg_L = 1.0
reg_R = 1.0

# Variables for calculating the MAE
loss_vals = []

sum = 0.0
sumL = 0.0
sumR = 0.0

# Assign initial guess for the L-BFGS-B
weights0 = np.zeros(28)

bestLoss = 1000000


for ep in range(n_epochs):
  
  print("#############")
  print("#  Epoch: ", ep+1)
  print("#############")

  # Set the collocation point
  x_in = tf.Variable(np.random.uniform(low=0.0, high=Lx, size=n_collocation_points)) # for random collocation points
  #x_in = tf.Variable(x_collocation_point) # for uniform dist. collocation points
  for (iip) in range (1):
    if eng.run_progs:
       eng.reset()

  

    # make a forward pass to calculate the residual function and calculate the loss function
    with tf.GradientTape() as tape3:
      with tf.GradientTape() as tape2:
         with tf.GradientTape() as tape1:
              result = eng.run(prog, args={"x": x_in, "alpha1": alpha1_in, "phi1": phi1_in, "r1": r1_in, "sq_r1": sq_r1_in, "sq_phi1": sq_phi1_in, "r2": r2_in, "kappa1": kappa1_in, \
                           "alpha1_2": alpha1_in_2, "phi1_2": phi1_in_2, "r1_2": r1_in_2, "sq_r1_2": sq_r1_in_2, "sq_phi1_2": sq_phi1_in_2, "r2_2": r2_in_2, "kappa1_2": kappa1_in_2, \
                           "alpha1_3": alpha1_in_3, "phi1_3": phi1_in_3, "r1_3": r1_in_3, "sq_r1_3": sq_r1_in_3, "sq_phi1_3": sq_phi1_in_3, "r2_3": r2_in_3, "kappa1_3": kappa1_in_3, \
                           "alpha1_4": alpha1_in_4, "phi1_4": phi1_in_4, "r1_4": r1_in_4, "sq_r1_4": sq_r1_in_4, "sq_phi1_4": sq_phi1_in_4, "r2_4": r2_in_4, "kappa1_4": kappa1_in_4})
            
              state = result.state
              #mean1, var1 = state.mean_photon(0)
              mean1, var1 = state.quad_expectation(0)
    
         dudx = tape1.gradient(mean1, x_in)
      du2dx2 = tape2.gradient(dudx, x_in)
      # define the known term
      b =  x_in*(x_in - 1)      
      #b =  1.0*(np.sin(2.0*x_in))

    
      # define the residual
      np_mean = mean1.numpy()
    
      res0 = du2dx2 - b
      differenceNew = tf.concat([ [ mean1[0] ] , res0[1:n_collocation_points-1]], 0) # add boundary one the left
      differenceNew = tf.concat([ differenceNew, [mean1[n_collocation_points-1]]], 0) # add boundary one the right
      
      #loss = tf.reduce_mean(tf.abs(differenceNew) ** 2 )
      loss = tf.reduce_mean(tf.abs(differenceNew))
      loss = tf.squeeze(tf.dtypes.cast(loss, tf.float32))
      #sum = sum + loss
      #loss = sum / (ep + 1)
     
      print("Inner Point Loss: ", loss.numpy())
  
  # LEFT Boundary
  
      if eng.run_progs:
         eng.reset()
      x_null = tf.Variable(np.zeros(n_collocation_points))
  
    
      result = eng.run(prog, args={"x": x_null, "alpha1": alpha1_in, "phi1": phi1_in, "r1": r1_in, "sq_r1": sq_r1_in, "sq_phi1": sq_phi1_in, "r2": r2_in, "kappa1": kappa1_in, \
                      "alpha1_2": alpha1_in_2, "phi1_2": phi1_in_2, "r1_2": r1_in_2, "sq_r1_2": sq_r1_in_2, "sq_phi1_2": sq_phi1_in_2, "r2_2": r2_in_2, "kappa1_2": kappa1_in_2, \
                      "alpha1_3": alpha1_in_3, "phi1_3": phi1_in_3, "r1_3": r1_in_3, "sq_r1_3": sq_r1_in_3, "sq_phi1_3": sq_phi1_in_3, "r2_3": r2_in_3, "kappa1_3": kappa1_in_3, \
                      "alpha1_4": alpha1_in_4, "phi1_4": phi1_in_4, "r1_4": r1_in_4, "sq_r1_4": sq_r1_in_4, "sq_phi1_4": sq_phi1_in_4, "r2_4": r2_in_4, "kappa1_4": kappa1_in_4})
            
      state = result.state
      mean0, var0 = state.quad_expectation(0)
      lossL = tf.reduce_mean(tf.abs(mean0))
      #sumL = sumL + lossL
      #lossL = sumL / ( ep +1)
        
   
  
      print("BC LEFT - Loss: ", lossL.numpy())


  # RIGHT Boundary
  #if (ep%1 ==0):
      if eng.run_progs:
           eng.reset()

      x_ones = tf.Variable(Lx*np.ones(n_collocation_points))
  
    #with tf.GradientTape() as tapeR:
      result = eng.run(prog, args={"x": x_ones, "alpha1": alpha1_in, "phi1": phi1_in, "r1": r1_in, "sq_r1": sq_r1_in, "sq_phi1": sq_phi1_in, "r2": r2_in, "kappa1": kappa1_in, \
                      "alpha1_2": alpha1_in_2, "phi1_2": phi1_in_2, "r1_2": r1_in_2, "sq_r1_2": sq_r1_in_2, "sq_phi1_2": sq_phi1_in_2, "r2_2": r2_in_2, "kappa1_2": kappa1_in_2, \
                      "alpha1_3": alpha1_in_3, "phi1_3": phi1_in_3, "r1_3": r1_in_3, "sq_r1_3": sq_r1_in_3, "sq_phi1_3": sq_phi1_in_3, "r2_3": r2_in_3, "kappa1_3": kappa1_in_3, \
                      "alpha1_4": alpha1_in_4, "phi1_4": phi1_in_4, "r1_4": r1_in_4, "sq_r1_4": sq_r1_in_4, "sq_phi1_4": sq_phi1_in_4, "r2_4": r2_in_4, "kappa1_4": kappa1_in_4})
            
      state = result.state
    # Note that all processing, including state-based post-processing,
    # must be done within the gradient tape context!
    #mean1, var1 = state.mean_photon(0)
      mean0, var0 = state.quad_expectation(0)
      lossR = tf.reduce_mean(tf.abs(mean0))
      #sumR = sumR + lossR
      #lossR = sumR / (ep +1)
      
      print("BC RIGHT - Loss: ", lossR.numpy())

      loss = loss + lossL + lossR
      print("Total loss: ", loss.numpy())


  # Back prop
  #if (ep < 100):
  #opt = tf.keras.optimizers.Adam(learning_rate=rate, name="Adam") 
  #else:
  #print("SGD")
  opt = tf.keras.optimizers.SGD(learning_rate=rate, name="SGD") 

  gradients = tape3.gradient(loss, [alpha1_in, phi1_in, r1_in, sq_r1_in, sq_phi1_in, r2_in, kappa1_in, \
                                      alpha1_in_2, phi1_in_2, r1_in_2, sq_r1_in_2, sq_phi1_in_2, r2_in_2, kappa1_in_2, \
                                      alpha1_in_3, phi1_in_3, r1_in_3, sq_r1_in_3, sq_phi1_in_3, r2_in_3, kappa1_in_3, \
                                      alpha1_in_4, phi1_in_4, r1_in_4, sq_r1_in_4, sq_phi1_in_4, r2_in_4, kappa1_in_4 ])
  opt.apply_gradients(zip(gradients, [alpha1_in, phi1_in, r1_in, sq_r1_in, sq_phi1_in, r2_in, kappa1_in, \
                                      alpha1_in_2, phi1_in_2, r1_in_2, sq_r1_in_2, sq_phi1_in_2, r2_in_2, kappa1_in_2, \
                                      alpha1_in_3, phi1_in_3, r1_in_3, sq_r1_in_3, sq_phi1_in_3, r2_in_3, kappa1_in_3, \
                                      alpha1_in_4, phi1_in_4, r1_in_4, sq_r1_in_4, sq_phi1_in_4, r2_in_4, kappa1_in_4 ]))
  
  loss_vals.append(loss)

  #if ((lossInner + lossR + lossL) < bestLoss):
  if ((loss ) < bestLoss):
    print("Best loss is recorded: ", (loss.numpy()))
    #print("Best loss is recorded: ", (lossInner + lossR + lossL))
    bestLoss = (loss)
    #bestLoss = (lossInner + lossR + lossL)
    
  
  if(True):
    #
    # First Layer
    weights0[0] = alpha1_in.numpy()
    weights0[1] = phi1_in.numpy()
    weights0[2] = r1_in.numpy()
    weights0[3] = sq_r1_in.numpy()
    weights0[4] = sq_phi1_in.numpy()
    weights0[5] = r2_in.numpy()
    weights0[6] = kappa1_in.numpy()

    # Second Layer
    weights0[7] = alpha1_in_2.numpy()
    weights0[8] = phi1_in_2.numpy()
    weights0[9] = r1_in_2.numpy()
    weights0[10] = sq_r1_in_2.numpy()
    weights0[11] = sq_phi1_in_2.numpy()
    weights0[12] = r2_in_2.numpy()
    weights0[13] = kappa1_in_2.numpy()

    # Third Layer
    weights0[14] = alpha1_in_3.numpy()
    weights0[15] = phi1_in_3.numpy()
    weights0[16] = r1_in_3.numpy()
    weights0[17] = sq_r1_in_3.numpy()
    weights0[18] = sq_phi1_in_3.numpy()
    weights0[19] = r2_in_3.numpy()
    weights0[20] = kappa1_in_3.numpy()

    # Fourth Layer
    weights0[21] = alpha1_in_4.numpy()
    weights0[22] = phi1_in_4.numpy()
    weights0[23] = r1_in_4.numpy()
    weights0[24] = sq_r1_in_4.numpy()
    weights0[25] = sq_phi1_in_4.numpy()
    weights0[26] = r2_in_4.numpy()
    weights0[27] = kappa1_in_4.numpy()

# Save before L-BFGS-B
# Save Loss
np.savetxt('LossFunction4L_Quad_R01_SGD_BS128.txt', np.c_[loss_vals] )


#####
# BFGS PART
#####


################ Inner Points 1: #####################
def model(weights): # here the weights become a tensor
    # make sure to reset the simulator
    if eng.run_progs:
       eng.reset()

    
    np_weights = weights.numpy()

    
    # first layer
    alpha1_in.assign(np_weights[0])
    phi1_in.assign(np_weights[1])
    r1_in.assign(np_weights[2])
    sq_r1_in.assign(np_weights[3])
    sq_phi1_in.assign(np_weights[4])
    r2_in.assign(np_weights[5])
    kappa1_in.assign(np_weights[6])

    # second layer
    alpha1_in_2.assign(np_weights[7])
    phi1_in_2.assign(np_weights[8])
    r1_in_2.assign(np_weights[9])
    sq_r1_in_2.assign(np_weights[10])
    sq_phi1_in_2.assign(np_weights[11])
    r2_in_2.assign(np_weights[12])
    kappa1_in_2.assign(np_weights[13])

    # third layer
    alpha1_in_3.assign(np_weights[14])
    phi1_in_3.assign(np_weights[15])
    r1_in_3.assign(np_weights[16])
    sq_r1_in_3.assign(np_weights[17])
    sq_phi1_in_3.assign(np_weights[18])
    r2_in_3.assign(np_weights[19])
    kappa1_in_3.assign(np_weights[20])

    # fourth layer
    alpha1_in_4.assign(np_weights[21])
    phi1_in_4.assign(np_weights[22])
    r1_in_4.assign(np_weights[23])
    sq_r1_in_4.assign(np_weights[24])
    sq_phi1_in_4.assign(np_weights[25])
    r2_in_4.assign(np_weights[26])
    kappa1_in_4.assign(np_weights[27])


    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            result = eng.run(prog, args={"x": x_in, "alpha1": alpha1_in, "phi1": phi1_in, "r1": r1_in, "sq_r1": sq_r1_in, "sq_phi1": sq_phi1_in, "r2": r2_in, "kappa1": kappa1_in,
                                                    "alpha1_2": alpha1_in_2, "phi1_2": phi1_in_2, "r1_2": r1_in_2, "sq_r1_2": sq_r1_in_2, "sq_phi1_2": sq_phi1_in_2, "r2_2": r2_in_2, "kappa1_2": kappa1_in_2,
                                                    "alpha1_3": alpha1_in_3, "phi1_3": phi1_in_3, "r1_3": r1_in_3, "sq_r1_3": sq_r1_in_3, "sq_phi1_3": sq_phi1_in_3, "r2_3": r2_in_3, "kappa1_3": kappa1_in_3,
                                                    "alpha1_4": alpha1_in_4, "phi1_4": phi1_in_4, "r1_4": r1_in_4, "sq_r1_4": sq_r1_in_4, "sq_phi1_4": sq_phi1_in_4, "r2_4": r2_in_4, "kappa1_4": kappa1_in_4})
            
            state = result.state
            # Note that all processing, including state-based post-processing,
            # must be done within the gradient tape context!
            #mean1, var1 = state.mean_photon(0)
            mean, _ = state.quad_expectation(0)
    
        dudx = tape1.gradient(mean, x_in)
    du2dx2 = tape2.gradient(dudx, x_in)

    
    # define the known term
    b = x_in*(x_in - 1)        
    #b = 1.0*(np.sin(2.0*x_in))
    
    
    res0 = du2dx2 - b
    #res0 = mean - b
    differenceNew = tf.concat([ [ mean[0] ] , res0[1:n_collocation_points-1]], 0) # add boundary one the left
    differenceNew = tf.concat([ differenceNew, [mean[n_collocation_points-1]]], 0) # add boundary one the right

      
    #loss = tf.reduce_mean(tf.abs(differenceNew) ** 2 )
    loss = tf.reduce_mean(tf.abs(differenceNew) )
    loss = tf.squeeze(tf.dtypes.cast(loss, tf.float32))

    # add BC - LEFT
    # make sure to reset the simulator
    if eng.run_progs:
       eng.reset()

    x_null = tf.Variable(np.zeros(n_collocation_points))
    result = eng.run(prog, args={"x": x_null, "alpha1": alpha1_in, "phi1": phi1_in, "r1": r1_in, "sq_r1": sq_r1_in, "sq_phi1": sq_phi1_in, "r2": r2_in, "kappa1": kappa1_in, \
                      "alpha1_2": alpha1_in_2, "phi1_2": phi1_in_2, "r1_2": r1_in_2, "sq_r1_2": sq_r1_in_2, "sq_phi1_2": sq_phi1_in_2, "r2_2": r2_in_2, "kappa1_2": kappa1_in_2, \
                      "alpha1_3": alpha1_in_3, "phi1_3": phi1_in_3, "r1_3": r1_in_3, "sq_r1_3": sq_r1_in_3, "sq_phi1_3": sq_phi1_in_3, "r2_3": r2_in_3, "kappa1_3": kappa1_in_3, \
                      "alpha1_4": alpha1_in_4, "phi1_4": phi1_in_4, "r1_4": r1_in_4, "sq_r1_4": sq_r1_in_4, "sq_phi1_4": sq_phi1_in_4, "r2_4": r2_in_4, "kappa1_4": kappa1_in_4})
            
    state = result.state
    # Note that all processing, including state-based post-processing,
    # must be done within the gradient tape context!
    #mean1, var1 = state.mean_photon(0)
    mean0, _ = state.quad_expectation(0)
    lossL = tf.reduce_mean(tf.abs(mean0))
    lossL = tf.squeeze(tf.dtypes.cast(lossL, tf.float32))

    if eng.run_progs:
       eng.reset()

    x_ones = tf.Variable(Lx*np.ones(n_collocation_points))
    result = eng.run(prog, args={"x": x_ones, "alpha1": alpha1_in, "phi1": phi1_in, "r1": r1_in, "sq_r1": sq_r1_in, "sq_phi1": sq_phi1_in, "r2": r2_in, "kappa1": kappa1_in, \
                      "alpha1_2": alpha1_in_2, "phi1_2": phi1_in_2, "r1_2": r1_in_2, "sq_r1_2": sq_r1_in_2, "sq_phi1_2": sq_phi1_in_2, "r2_2": r2_in_2, "kappa1_2": kappa1_in_2, \
                      "alpha1_3": alpha1_in_3, "phi1_3": phi1_in_3, "r1_3": r1_in_3, "sq_r1_3": sq_r1_in_3, "sq_phi1_3": sq_phi1_in_3, "r2_3": r2_in_3, "kappa1_3": kappa1_in_3, \
                      "alpha1_4": alpha1_in_4, "phi1_4": phi1_in_4, "r1_4": r1_in_4, "sq_r1_4": sq_r1_in_4, "sq_phi1_4": sq_phi1_in_4, "r2_4": r2_in_4, "kappa1_4": kappa1_in_4})
            
    state = result.state
    mean1, _ = state.quad_expectation(0)
    lossR = tf.reduce_mean(tf.abs(mean1))
    lossR = tf.squeeze(tf.dtypes.cast(lossR, tf.float32))


    loss = loss + lossL + lossR

   
    
    return(loss)


@tf.function
def val_and_grad(weights):
    with tf.GradientTape() as tape:
        tape.watch(weights)
        loss = model(weights)  #modelDF
    grad = tape.gradient(loss, [alpha1_in, phi1_in, r1_in, sq_r1_in, sq_phi1_in, r2_in, kappa1_in,
                                alpha1_in_2, phi1_in_2, r1_in_2, sq_r1_in_2, sq_phi1_in_2, r2_in_2, kappa1_in_2,
                                alpha1_in_3, phi1_in_3, r1_in_3, sq_r1_in_3, sq_phi1_in_3, r2_in_3, kappa1_in_3,
                                alpha1_in_4, phi1_in_4, r1_in_4, sq_r1_in_4, sq_phi1_in_4, r2_in_4, kappa1_in_4])
    grad = tf.convert_to_tensor(grad)
    return loss, grad

def func(weights):
    vv_result = [vv.numpy().astype(np.float64)  for vv in val_and_grad(tf.constant(weights, dtype=tf.float32))]
    return vv_result



def predict(np_weights):
  # make sure to reset the simulator
    if eng.run_progs:
       eng.reset()

    x_in = tf.Variable(x_collocation_point) 
    
    # first layer
    alpha1_in.assign(np_weights[0])
    phi1_in.assign(np_weights[1])
    r1_in.assign(np_weights[2])
    sq_r1_in.assign(np_weights[3])
    sq_phi1_in.assign(np_weights[4])
    r2_in.assign(np_weights[5])
    kappa1_in.assign(np_weights[6])

    # second layer
    alpha1_in_2.assign(np_weights[7])
    phi1_in_2.assign(np_weights[8])
    r1_in_2.assign(np_weights[9])
    sq_r1_in_2.assign(np_weights[10])
    sq_phi1_in_2.assign(np_weights[11])
    r2_in_2.assign(np_weights[12])
    kappa1_in_2.assign(np_weights[13])

    # third layer
    alpha1_in_3.assign(np_weights[14])
    phi1_in_3.assign(np_weights[15])
    r1_in_3.assign(np_weights[16])
    sq_r1_in_3.assign(np_weights[17])
    sq_phi1_in_3.assign(np_weights[18])
    r2_in_3.assign(np_weights[19])
    kappa1_in_3.assign(np_weights[20])

    # fourth layer
    alpha1_in_4.assign(np_weights[21])
    phi1_in_4.assign(np_weights[22])
    r1_in_4.assign(np_weights[23])
    sq_r1_in_4.assign(np_weights[24])
    sq_phi1_in_4.assign(np_weights[25])
    r2_in_4.assign(np_weights[26])
    kappa1_in_4.assign(np_weights[27])


    result = eng.run(prog, args={"x": x_in, "alpha1": alpha1_in, "phi1": phi1_in, "r1": r1_in, "sq_r1": sq_r1_in, "sq_phi1": sq_phi1_in, "r2": r2_in, "kappa1": kappa1_in, \
                      "alpha1_2": alpha1_in_2, "phi1_2": phi1_in_2, "r1_2": r1_in_2, "sq_r1_2": sq_r1_in_2, "sq_phi1_2": sq_phi1_in_2, "r2_2": r2_in_2, "kappa1_2": kappa1_in_2, \
                      "alpha1_3": alpha1_in_3, "phi1_3": phi1_in_3, "r1_3": r1_in_3, "sq_r1_3": sq_r1_in_3, "sq_phi1_3": sq_phi1_in_3, "r2_3": r2_in_3, "kappa1_3": kappa1_in_3, \
                      "alpha1_4": alpha1_in_4, "phi1_4": phi1_in_4, "r1_4": r1_in_4, "sq_r1_4": sq_r1_in_4, "sq_phi1_4": sq_phi1_in_4, "r2_4": r2_in_4, "kappa1_4": kappa1_in_4})
            
    state = result.state
    mean1, _ = state.quad_expectation(0)
    plt.plot(mean1)
    plt.show()
    np.savetxt('Res4L_Quad_R01_SGD_BS128.txt', mean1.numpy())
    np.savetxt('Weights4L_Quad_R01_SGD_128.txt', np_weights)

predict(weights0)


#x_in = tf.Variable(x_collocation_point) 
#x_in = tf.Variable(np.random.uniform(low=0.0, high=Lx, size=n_collocation_points)) 
#resdd= sopt.minimize(fun=func, x0=weights0, tol=1E-9, jac=True, method='L-BFGS-B',  options={'disp': True, 'maxiter': 50})
#weights0 = resdd.x
#predict(weights0)
#print("Final Loss: ", resdd.fun)
#weights0 = resdd.x
#print("Loss: ", resdd.fun)
#predict(weights0)