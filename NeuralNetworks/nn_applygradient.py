import numpy as np

def nn_applygradient(nn):
    method = nn.optimization_method
    if method == 'AdaGrad' or method == 'RMSProp' or method == 'Adam':
        grad_squared = 0
        if nn.batch_normalization == 0:
            for k in range(nn.depth-1):
                grad_squared = grad_squared + sum(sum(nn.W_grad[k]**2)) + sum(nn.b_grad[k]**2)
        else:
            for k in range(nn.depth-1):
                grad_squared = grad_squared + sum(sum(nn.W_grad[k]**2)) + sum(nn.b_grad[k]**2) + nn.Gamma[k]**2 + nn.Beta[k]**2

    if method == 'Adam':
        nn.AdamTime +=1


    for k in range(nn.depth-1):
        if nn.batch_normalization == 0:
            if method == 'normal':
                nn.W[k] = nn.W[k] - nn.learning_rate*nn.W_grad[k]
                nn.b[k] = nn.b[k] - nn.learning_rate*nn.b_grad[k]
                
            elif method == 'AdaGrad':
                nn.rW[k] = nn.rW[k] + nn.W_grad[k]**2
                nn.rb[k] = nn.rb[k] + nn.b_grad[k]**2
                nn.W[k] = nn.W[k] - nn.learning_rate*nn.W_grad[k]/(np.sqrt(nn.rW[k]) + 0.001)
                nn.b[k] = nn.b[k] - nn.learning_rate*nn.b_grad[k]/(np.sqrt(nn.rb[k]) + 0.001)
                
            elif method == 'Momentum':
                rho = 0.1 #rho = 0.1
                nn.vW[k] = rho * nn.vW[k] - nn.learning_rate*nn.W_grad[k]
                nn.vb[k] = rho * nn.vb[k] - nn.learning_rate*nn.b_grad[k]
                nn.W[k] = nn.W[k] + nn.vW[k]
                nn.b[k] = nn.b[k] + nn.vb[k]

            elif method == 'RMSProp':
                rho = 0.9 #rho = 0.9
                nn.rW[k] = rho * nn.rW[k] + 0.1*nn.W_grad[k]**2
                nn.rb[k] = rho * nn.rb[k] + 0.1*nn.b_grad[k]**2
                nn.W[k] = nn.W[k] - nn.learning_rate*nn.W_grad[k]/(np.sqrt(nn.rW[k]) + 0.001)
                nn.b[k] = nn.b[k] - nn.learning_rate*nn.b_grad[k]/(np.sqrt(nn.rb[k]) + 0.001) #rho = 0.9

            elif method == 'Adam':
                rho1 = 0.9
                rho2 = 0.999
                nn.sW[k] = 0.9*nn.sW[k] + 0.1*nn.W_grad[k]
                nn.sb[k] = 0.9*nn.sb[k] + 0.1*nn.b_grad[k]
                nn.rW[k] = 0.999*nn.rW[k] + 0.001*nn.W_grad[k]**2
                nn.rb[k] = 0.999*nn.rb[k] + 0.001*nn.b_grad[k]**2

                newS = nn.sW[k] / (1 - rho1)
                newR = nn.rW[k] / (1 - rho2)
                nn.W[k] = nn.W[k] - nn.learning_rate*newS/np.sqrt(newR + 0.00001)
                newS = nn.sb[k] / (1 - rho1)
                newR = nn.rb[k] / (1 - rho2)
                nn.b[k] = nn.b[k] -nn.learning_rate*newS/np.sqrt(newR + 0.00001)#rho1 = 0.9, rho2 = 0.999, delta = 0.00001

        else:
            if method == 'normal':
                nn.W[k] = nn.W[k] - nn.learning_rate*nn.W_grad[k]
                nn.b[k] = nn.b[k] - nn.learning_rate*nn.b_grad[k]
                nn.Gamma[k] = nn.Gamma[k] - nn.learning_rate*nn.Gamma_grad[k]
                nn.Beta[k] = nn.Beta[k] - nn.learning_rate*nn.Beta_grad[k]
                
            elif method == 'AdaGrad':
                nn.rW[k] = nn.rW[k] + nn.W_grad[k]**2
                nn.rb[k] = nn.rb[k] + nn.b_grad[k]**2
                nn.rGamma[k] = nn.rGamma[k] + nn.Gamma_grad[k]**2
                nn.rBeta[k] = nn.rBeta[k] + nn.Beta_grad[k]**2
                nn.W[k] = nn.W[k] - nn.learning_rate*nn.W_grad[k]/(np.sqrt(nn.rW[k]) + 0.001)
                nn.b[k] = nn.b[k] - nn.learning_rate*nn.b_grad[k]/(np.sqrt(nn.rb[k]) + 0.001)
                nn.Gamma[k] = nn.Gamma[k] - nn.learning_rate*nn.Gamma_grad[k] / (np.sqrt(nn.rGamma[k]) + 0.001)
                nn.Beta[k] = nn.Beta[k] - nn.learning_rate*nn.Beta_grad[k] / (np.sqrt(nn.rBeta[k]) + 0.001)
                
            elif method == 'RMSProp':
                nn.rW[k] = 0.9*nn.rW[k] + 0.1*nn.W_grad[k]**2
                nn.rb[k] = 0.9*nn.rb[k] + 0.1*nn.b_grad[k]**2
                nn.rGamma[k] = 0.9*nn.rGamma[k] + 0.1*nn.Gamma_grad[k]**2
                nn.rBeta[k] = 0.9*nn.rBeta[k] + 0.1*nn.Beta_grad[k]**2
                nn.W[k] = nn.W[k] - nn.learning_rate*nn.W_grad[k]/(np.sqrt(nn.rW[k]) + 0.001)
                nn.b[k] = nn.b[k] - nn.learning_rate*nn.b_grad[k]/(np.sqrt(nn.rb[k]) + 0.001)
                nn.Gamma[k] = nn.Gamma[k] - nn.learning_rate*nn.Gamma_grad[k] / (np.sqrt(nn.rGamma[k]) + 0.001)
                nn.Beta[k] = nn.Beta[k] - nn.learning_rate*nn.Beta_grad[k] / (np.sqrt(nn.rBeta[k]) + 0.001) #rho = 0.9

            elif method == 'Momentum':
                rho = 0.1 #rho = 0.1
                nn.vW[k] = rho * nn.vW[k] - nn.learning_rate*nn.W_grad[k]
                nn.vb[k] = rho * nn.vb[k] - nn.learning_rate*nn.b_grad[k]
                nn.vGamma[k] = rho * nn.vGamma[k] - nn.learning_rate*nn.Gamma_grad[k]
                nn.vBeta[k] = rho * nn.vBeta[k] - nn.learning_rate*nn.Beta_grad[k]
                nn.W[k] = nn.W[k] + nn.vW[k]
                nn.b[k] = nn.b[k] + nn.vb[k]
                nn.Gamma[k] = nn.Gamma[k] + nn.vGamma[k]
                nn.Beta[k] = nn.Beta[k] + nn.vBeta[k]

            elif method == 'Adam':
                rho1 = 0.9;
                rho2 = 0.999;
                nn.sW[k] = rho1*nn.sW[k] + (1-rho1)*nn.W_grad[k]
                nn.sb[k] = rho1*nn.sb[k] + (1-rho1)*nn.b_grad[k]
                nn.sGamma[k] = rho1*nn.sGamma[k] + (1-rho1)*nn.Gamma_grad[k]
                nn.sBeta[k] = rho1*nn.sBeta[k] + (1-rho1)*nn.Beta_grad[k]
                nn.rW[k] = rho2*nn.rW[k] + (1-rho2)*nn.W_grad[k]**2
                nn.rb[k] = rho2*nn.rb[k] + (1-rho2)*nn.b_grad[k]**2
                nn.rBeta[k] = rho2*nn.rBeta[k] + (1-rho2)*nn.Beta_grad[k]**2
                nn.rGamma[k] = rho2*nn.rGamma[k] + (1-rho2)*nn.Gamma_grad[k]**2

                newS = nn.sW[k]/(1 - rho1**nn.AdamTime)
                newR = nn.rW[k]/(1 - rho2**nn.AdamTime)
                nn.W[k] = nn.W[k]-nn.learning_rate * newS/np.sqrt(newR + 0.00001)
                newS = nn.sb[k]/(1 - rho1**nn.AdamTime)
                newR = nn.rb[k]/(1 - rho2**nn.AdamTime)
                nn.b[k] = nn.b[k] -nn.learning_rate * newS/np.sqrt(newR + 0.00001)
                newS = nn.sGamma[k] / (1 - rho1 ** nn.AdamTime)
                newR = nn.rGamma[k] / (1 - rho2 ** nn.AdamTime)
                nn.Gamma[k] = nn.Gamma[k] - nn.learning_rate*newS/np.sqrt(newR + 0.00001)
                newS = nn.sBeta[k] / (1 - rho1 ** nn.AdamTime)
                newR = nn.rBeta[k] / (1 - rho2 ** nn.AdamTime)
                nn.Beta[k] = nn.Beta[k] -nn.learning_rate*newS/np.sqrt(newR + 0.00001)
    return nn