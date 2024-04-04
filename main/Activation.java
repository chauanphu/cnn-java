public abstract class Activation {
    Tensor2D cached_z;
    Activation() {}
    
    /**
     * Calculate the activation function
     * @param tensor the input is z, a = activation(z); z's dimensions = [n, m], a's dimensions = [n, m]
     * @return
     */
    public abstract Tensor2D cal(Tensor2D tensor);
    
    /**
     * Calculate the derivative of the activation function
     * @param tensor the input is z, a = activation(z); z's dimensions = [n, m], a's dimensions = [n, m]
     * @return the derivative of the activation function (dz)
     */
    public abstract Tensor2D derivative(Tensor2D tensor);
    
    public static void main(String[]args){
        Tensor2D tensor = new Tensor2D(new double[][]{
            {1, 2, 3},
            {0, 5, 6},
            {-2, 0, 9}
        });
        Activation activation = new Relu();
        System.out.println(activation.cal(tensor));
    }
}

class Relu extends Activation {
    Relu() {}

    @Override
    public Tensor2D cal(Tensor2D tensor) {
        Tensor2D res = new Tensor2D(tensor.nrows, tensor.ncols);
        for (int i = 0; i < tensor.nrows; i++) {
            for (int j = 0; j < tensor.ncols; j++) {
                res.data[i][j] = Math.max(0, tensor.data[i][j]);
            }
        }
        return res;
    }
    
    @Override
    public Tensor2D derivative(Tensor2D tensor) {
        Tensor2D res = new Tensor2D(tensor.nrows, tensor.ncols);
        for (int i = 0; i < tensor.nrows; i++) {
            for (int j = 0; j < tensor.ncols; j++) {
                res.data[i][j] = tensor.data[i][j] > 0 ? 1 : 0;
            }
        }
        return res;
    }
}

class Softmax extends Activation {
    Softmax() {}

    @Override
    public Tensor2D cal(Tensor2D tensor) {
        // tensor's dimension: (feature, sample)
        Tensor2D res = new Tensor2D(tensor.nrows, tensor.ncols);

        // Iterate through each sample
        for (int i = 0; i < tensor.ncols; i++) {
            double sum = 0;
            // calculate the sum in each sample
            for (int j = 0; j < tensor.nrows; j++) {
                // t = e^(z)
                double t = Math.exp(tensor.data[j][i]);
                sum += t;
            }
            
            for (int j = 0; j < tensor.nrows; j++) {
                double t = Math.exp(tensor.data[j][i]);
                double a = t / sum;
                res.data[j][i] = a;
            }
        }
        return res;
    }

    @Override
    public Tensor2D derivative(Tensor2D a) {
        Tensor2D res = new Tensor2D(a.nrows, a.ncols);
        for (int i = 0; i < a.nrows; i++) {
            for (int j = 0; j < a.ncols; j++) {
                for (int k = 0; k < a.ncols; k++) {
                    if (j == k) {
                        res.data[i][j] += a.data[i][k] * (1 - a.data[i][k]);
                    } else {
                        res.data[i][j] -= a.data[i][j] * a.data[i][k];
                    }
                }
            }
        }
        return res;
    }
}