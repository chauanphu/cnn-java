public class Activation {
    public static enum ActivationType {
        RELU, SIGMOID, SOFTMAX
    }
    private ActivationType activationType;
    Activation(ActivationType activationType) {
        this.activationType = activationType;
    }
    public Tensor2D cal(Tensor2D tensor) {
        switch (activationType) {
            case RELU:
                return relu(tensor);
            // case SIGMOID:
            //     return sigmoid(tensor);
            case SOFTMAX:
                return softmax(tensor);
            // Linear
            default:
                return tensor;
        }
    }
    private Tensor2D relu(Tensor2D tensor) {
        Tensor2D res = new Tensor2D(tensor.nrows, tensor.ncols);
        for (int i = 0; i < tensor.nrows; i++) {
            for (int j = 0; j < tensor.ncols; j++) {
                res.data[i][j] = Math.max(0, tensor.data[i][j]);
            }
        }
        return res;
    }

    private Tensor2D softmax(Tensor2D tensor) {
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

    public static void main(String[]args){
        Tensor2D tensor = new Tensor2D(new double[][]{
            {1, 2, 3},
            {0, 5, 6},
            {-2, 8, 9}
        });
        Activation activation = new Activation(ActivationType.SOFTMAX);
        System.out.println(activation.cal(tensor).argmax());
    }
}